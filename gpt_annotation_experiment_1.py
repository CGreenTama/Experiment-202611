"""
JTB 観光資源分類 — GPT 概念抽出スクリプト(施設・条件は事前定義)
=================================================================
ガイドライン:
  S ... プロンプトのみ (ガイドラインなし)
  G ... ガイドライン付き

モデル:
  gpt-5.6-terra  (非推論・軽量)  reasoning.effort=low
  gpt-5.6-terra  (推論・軽量)   reasoning.effort=low
  gpt-5.6-sol    (非推論・高性能)  reasoning.effort=low
  gpt-5.6-sol    (推論・高性能)  reasoning.effort=high

対象施設は TARGET_Facility、実験条件(Terra-S, Sol-G など)は CONDITIONS に
定義する。どちらも 1 件以上、任意件数。
対象: 施設数 × 条件数 × N_RUNS 回の API 呼び出し

結果カラム名(1 回の API 呼び出しごとに 1 セル。日時を含めないので中断後も同じ名前で再開できる):
  {モデル}_{推論}_{S/G}_run{回数}  付与ラベル
    例: gpt-5.6-terra_low_S_run1
  Labelingreason-{上記カラム名}   付与理由
  推論(effort)は、Terra-S と Terra-H-S のようにモデルと S/G が同じ条件を区別するために含める。

途中からの再開(RESUME = True):
  最新の結果CSVに今回のカラムの未記入セルがあれば、その内容を引き継いで続きを実行する。
  記入済みのセルは呼び出さない。API エラーのセルはラベルを空欄のまま残すので、再実行でやり直す。
  結果CSV・料金CSV・時間CSVは 1 回呼び出すごとに保存する。
  未記入セルがない場合や、今回のカラムを 1 つも含まない場合(旧形式など)は新しい実験として始める。
  RESUME = False なら常に新しい実験として始める。

出力ファイル(OUTPUT_DIR に保存):
  実行するたびに、その実行の開始日時 YYYYMMDD_HHMM を付けた 3 ファイルを作る。
  再開した場合も今回の日時のファイルになり、再開元の結果CSVは変更しない。
  そのため最新の結果CSVに、それまでの全結果が入る。experiment 列も今回の日時。
  結果CSV  *_gpt_results_{日時}.csv  ラベルと付与理由(再開時は再開元の内容を引き継ぐ)
    例: yokohama_label_comparison_7_gpt_results_20261105_1230.csv
  料金CSV  *_gpt_cost_{日時}.csv     今回の実行の、呼び出しごとのトークン数と料金
    cost_usd = (input_tokens - cached_tokens) × Input 単価
             + cached_tokens × Cached input 単価 + output_tokens × Output 単価
  時間CSV  *_gpt_time_{日時}.csv     今回の実行の、呼び出しごとの開始・終了・経過時間
    start   : 開始時刻
    end     : 終了時刻
    elapsed : 経過時間(秒) = end - start
    呼び出しの行は SLEEP を含まない。
    condition=ALL の行は今回の実行全体の時間で、SLEEP を含む。

使い方:
  pip install openai pandas python-dotenv

  .env に OPENAI_API_KEY=sk-... を記載
  python gpt_annotation_experiment_1.py
=================================================================
"""

import os, json, re, ast, time, glob
import pandas as pd
from dotenv import load_dotenv
import openai
from datetime import datetime
from pathlib import Path

env_path = Path.home() / ".env"
load_dotenv(dotenv_path=env_path)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ================================================================
# 0. 設定
# ================================================================
CSV_PATH    = "yokohama_label_comparison_7.csv"
GUIDELINE_G = "JTB_付与規則_IF-THEN-EXCEPT_112.txt"
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output112")

N_RUNS    = 10     # 同一条件の繰り返し回数(分散確認用)
SLEEP      = 1.0    # API呼び出し間のウェイト(秒)
MAX_TOKENS = 2000
RESUME     = True   # True: 最新の結果CSVに未記入セルがあれば続きから再開 / False: 常に新しい実験

# 対象施設(1件以上、任意件数)
TARGET_Facility = [
    "横浜開港資料館", "野毛山動物園"
]

# 実験条件(1件以上、任意件数)  guideline: S=なし / G=あり
CONDITIONS = [
    dict(label="Terra-S", model="gpt-5.6-terra", effort="low",  guideline="S"),
    dict(label="Terra-G", model="gpt-5.6-terra", effort="low",  guideline="G"),
    dict(label="Sol-S",   model="gpt-5.6-sol",   effort="high", guideline="S"),
    dict(label="Sol-G",   model="gpt-5.6-sol",   effort="high", guideline="G"),
    dict(label="Terra-H-S", model="gpt-5.6-terra", effort="high", guideline="S"),
    dict(label="Terra-H-G", model="gpt-5.6-terra", effort="high", guideline="G"),
    dict(label="Sol-L-S",   model="gpt-5.6-sol",   effort="low",  guideline="S"),
    dict(label="Sol-L-G",   model="gpt-5.6-sol",   effort="low",  guideline="G"),
]

# 料金($/100万トークン) — 2026年7月30日改定後。Input / Cached input / Output 区分
# cached_input は _review.py と同じく Input の 10% を仮置き。料金表で要確認。
PRICE = {
    "gpt-5.6-terra": dict(input=2.50, cached_input=0.25, output=15.00),
    "gpt-5.6-sol":   dict(input=5.00, cached_input=0.50, output=30.00),
}

# ================================================================
# 1. カテゴリ定義 — 日本語表記に統一
# ================================================================
CATEGORIES = [
    "山岳", "高原・湿原・原野", "湖沼", "河川・峡谷", "滝",
    "海岸・岬", "岩石・洞窟", "動物", "植物", "自然現象",
    "史跡", "神社・寺院・教会", "城跡・城郭・宮殿", "集落・街", "郷土景観",
    "庭園・公園", "建造物", "年中行事", "動植物園・水族館", "博物館・美術館",
    "テーマ公園・テーマ施設", "温泉", "食", "芸能・スポーツ", "その他",
]
CATEGORIES_STR = "、".join(CATEGORIES)

# ================================================================
# 2. プロンプト
# ================================================================
SYSTEM_S = f"""あなたはJTB観光資源分類の専門アノテーターです。
施設の説明文のみを根拠にラベルを付与してください。

【出力形式】
{{"labels": ["ラベル1", "ラベル2"], "reason": "付与理由"}}  ← JSON のみ。JSON の外に文章を書かない。

【利用可能なラベル】
{CATEGORIES_STR}

【原則】
- 説明文に書かれていない情報(外部知識)は使わない
- 名称だけを根拠にしない
- 該当なしの場合は「その他」を返す
- 複数該当する場合は複数返してよい
- reason には、なぜそのラベルを付与したかを、説明文のどの記述に基づくかを示して日本語の自然文で書く(複数ラベルの場合はラベルごとに書く)"""

# 付与規則の第5章は JSON のキー・識別子・理由の扱いが上記の【出力形式】と異なるため、
# 規則の後ろで適用しないことを明示する
SYSTEM_G_TEMPLATE = SYSTEM_S + """

【付与規則】
{GUIDELINE_PLACEHOLDER}

【出力に関する優先事項】
付与規則の「5. 出力形式」は適用せず、冒頭の【出力形式】に従うこと。
- キーは types ではなく labels と reason を使う
- 識別子(CUL_GARDEN など)ではなく【利用可能なラベル】の日本語表記で出力する
- 該当なしの場合は空配列ではなく「その他」を返す
- reason は省略せず必ず書く"""


def build_user(name, desc_ja, desc_en=""):
    msg = f"施設名: {name}\n説明文(日本語): {desc_ja}"
    if desc_en and str(desc_en) not in ("", "情報なし", "nan"):
        msg += f"\n説明文(英語): {desc_en}"
    msg += "\n\nJSON のみで回答:"
    return msg


# ================================================================
# 3. API 呼び出し
# ================================================================
def call_gpt(model_str, effort, system_text, user_text):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    resp = client.responses.create(
        model=model_str,
        reasoning={"effort": effort},
        instructions=system_text,
        input=user_text,
        max_output_tokens=MAX_TOKENS,
    )
    # レスポンステキスト取得
    output_text = ""
    for item in resp.output:
        if hasattr(item, "content"):
            for block in item.content:
                if hasattr(block, "text"):
                    output_text += block.text

    # トークン数取得(cached_tokens は input_tokens のうちキャッシュ適用分)
    usage = resp.usage
    in_tokens  = getattr(usage, "input_tokens",  0)
    out_tokens = getattr(usage, "output_tokens", 0)
    in_details = getattr(usage, "input_tokens_details", None)
    cached_tokens = (getattr(in_details, "cached_tokens", 0) or 0) if in_details else 0

    return output_text, in_tokens, cached_tokens, out_tokens


def calc_cost(model_str, in_tokens, cached_tokens, out_tokens):
    """料金(USD) = 非キャッシュ入力 × Input + キャッシュ入力 × Cached input + 出力 × Output"""
    p = PRICE[model_str]
    return ((in_tokens - cached_tokens) / 1_000_000 * p["input"]
            + cached_tokens            / 1_000_000 * p["cached_input"]
            + out_tokens               / 1_000_000 * p["output"])


def parse_response(text):
    # reason の文中に } や改行が入っても JSON 全体を読めるようにする
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(), strict=False)
            labels = obj.get("labels", [])
            return [l for l in labels if l in CATEGORIES], str(obj.get("reason", ""))
        except json.JSONDecodeError:
            pass
    return [], ""


def fmt_time(dt):
    return dt.isoformat(sep=" ", timespec="milliseconds")


def elapsed_sec(start, end):
    """実行時間(秒) = 終了時刻 - 開始時刻"""
    return round((end - start).total_seconds(), 3)


# ================================================================
# 3-2. 結果CSV の読み書き(再開用)
# ================================================================
def label_col(cfg, run):
    """付与ラベルのカラム名: {モデル}_{推論}_{S/G}_run{回数}"""
    return f"{cfg['model']}_{cfg['effort']}_{cfg['guideline']}_run{run}"


def is_blank(value):
    return pd.isna(value) or str(value).strip() == ""


def select_targets(df):
    labeled = df[df["human_label"].notna() & (df["human_label"] != "")]
    return labeled[labeled["施設名"].isin(TARGET_Facility)]


def save_csv(df, path):
    """書き込み途中で中断してもファイルが壊れないよう、一時ファイル経由で置き換える"""
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def append_row(path, row):
    """CSV に 1 行追記する(整数が小数表記にならないよう文字列のまま結合する)"""
    new = pd.DataFrame([row]).astype(str)
    if os.path.exists(path):
        new = pd.concat([pd.read_csv(path, dtype=str), new], ignore_index=True)
    save_csv(new, path)


def find_resume_csv(stem, planned):
    """最新の結果CSVに今回のカラムの未記入セルがあれば、そのパスを返す"""
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{stem}_gpt_results_*.csv")))
    if not files:
        return None
    df = pd.read_csv(files[-1], dtype=str)
    if not any(c in df.columns for c in planned):
        return None          # 旧形式(カラム名に日時あり)や別条件の結果からは再開しない
    rows = select_targets(df).index
    pending = any(c not in df.columns or any(is_blank(df.at[i, c]) for i in rows)
                  for c in planned)
    return files[-1] if pending else None


# ================================================================
# 4. ガイドライン読み込み
# ================================================================
def load_guideline(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ガイドラインファイルが見つかりません: {path}\n"
            f"  → G条件はガイドライン本文が必須です。パスを確認してください。"
        )
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
        if not text.strip():
            raise ValueError(
                f"ガイドラインファイルの中身が空です: {path}\n"
                f"  → G条件のプロンプトが空のまま実行されるのを防ぐため停止しました。"
            )
    return text


# ================================================================
# 5. メイン実験ループ
# ================================================================
def run_experiment():
    if not TARGET_Facility:
        raise ValueError("TARGET_Facility に施設名を 1 件以上指定してください。")

    # 今回の実行の日時。再開した場合も含め、3 ファイルの名前と experiment 列に使う
    exp_start  = datetime.now()
    experiment = exp_start.strftime("%Y%m%d_%H%M")

    # 今回のカラム(日時を含めないので、中断後も同じ名前で再開できる)
    planned = [label_col(cfg, run) for cfg in CONDITIONS for run in range(1, N_RUNS + 1)]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem     = os.path.basename(CSV_PATH).replace(".csv", "")
    out_csv  = os.path.join(OUTPUT_DIR, f"{stem}_gpt_results_{experiment}.csv")
    cost_csv = os.path.join(OUTPUT_DIR, f"{stem}_gpt_cost_{experiment}.csv")
    time_csv = os.path.join(OUTPUT_DIR, f"{stem}_gpt_time_{experiment}.csv")

    # 再開できる結果CSVがあればその内容を引き継ぎ、なければ入力CSVから新しい実験として始める
    # (再開元のファイルは変更せず、今回の日時の結果CSVに保存する)
    resume_csv = find_resume_csv(stem, planned) if RESUME else None
    if resume_csv:
        df = pd.read_csv(resume_csv, dtype=str)
        print(f"再開元: {resume_csv}")
    else:
        df = pd.read_csv(CSV_PATH, dtype=str)
        print("新規の実験")
    print(f"保存先: {out_csv}")

    target_df = select_targets(df)
    missing = [n for n in TARGET_Facility if n not in set(target_df["施設名"])]
    if missing:
        print(f"[WARN] CSV に見つからない(または human_label 未設定の)施設: {missing}")
    if target_df.empty:
        raise ValueError("対象施設が 0 件です。TARGET_Facility と CSV を確認してください。")

    # まだ無いカラムを空欄で追加して保存する(未記入のセルが再開時の対象になる)
    cols = [c for col in planned for c in (col, f"Labelingreason-{col}")]
    add  = [c for c in cols if c not in df.columns]
    if add:
        df = pd.concat([df, pd.DataFrame(index=df.index, columns=add, dtype=object)], axis=1)
    save_csv(df, out_csv)

    n_total = len(target_df) * len(planned)
    n_todo  = sum(is_blank(df.at[i, c]) for c in planned for i in target_df.index)
    print(f"対象施設: {len(target_df)} 件")
    print(f"  {list(target_df['施設名'])}")
    print(f"API呼び出し予定: {len(target_df)}施設 × {len(CONDITIONS)}条件 × "
          f"{N_RUNS}回 = {n_total} 回 (記入済み {n_total - n_todo} 回 / 残り {n_todo} 回)\n")

    systems = {"S": SYSTEM_S}
    if any(cfg["guideline"] == "G" for cfg in CONDITIONS):
        guideline_text = load_guideline(GUIDELINE_G)
        systems["G"] = SYSTEM_G_TEMPLATE.replace("{GUIDELINE_PLACEHOLDER}", guideline_text)

    # コスト集計用(今回の実行分)
    total_cost = 0.0
    cost_log = []

    try:
        for cfg in CONDITIONS:
            system_text = systems[cfg["guideline"]]

            print(f"{'='*60}")
            print(f"条件   : {cfg['label']}"
                  f"  ({'ガイドラインなし' if cfg['guideline'] == 'S' else 'ガイドラインあり'})")
            print(f"モデル : {cfg['model']}  (reasoning.effort={cfg['effort']})")
            print(f"{'='*60}")

            for run in range(1, N_RUNS + 1):
                col_name = label_col(cfg, run)
                todo = [i for i in target_df.index if is_blank(df.at[i, col_name])]
                if not todo:
                    continue     # 記入済み
                print(f"\n--- run {run}/{N_RUNS} ---")

                run_in     = 0
                run_cached = 0
                run_out    = 0
                run_cost   = 0.0

                for idx in todo:
                    row     = df.loc[idx]
                    name    = row["施設名"]
                    desc_ja = str(row["editorialsummary_ja"])
                    desc_en = str(row.get("editorialSummary", ""))

                    user_text = build_user(name, desc_ja, desc_en)

                    call_start = datetime.now()
                    try:
                        raw, in_tok, cached_tok, out_tok = call_gpt(
                            cfg["model"], cfg["effort"],
                            system_text, user_text
                        )
                        call_end = datetime.now()
                        pred, reason = parse_response(raw)
                        df.at[idx, col_name] = str(pred)

                        # 結果を表示
                        print(f"  [{name}]")
                        print(f"    応答テキスト : {raw.strip()}")
                        print(f"    抽出ラベル   : {pred}")
                        print(f"    付与理由     : {reason}")
                        print(f"    トークン     : 入力={in_tok} (うちキャッシュ={cached_tok})  出力={out_tok}")
                        print(f"    実行時間     : {elapsed_sec(call_start, call_end)} 秒")

                    except Exception as e:
                        # ラベルは空欄のまま残し、再実行時にやり直す
                        call_end = datetime.now()
                        print(f"  [ERROR] {name}: {e}")
                        in_tok = cached_tok = out_tok = 0
                        reason = f"[ERROR] {e}"

                    df.at[idx, f"Labelingreason-{col_name}"] = reason
                    save_csv(df, out_csv)

                    # 料金・時間を 1 回ごとに追記
                    call_cost = calc_cost(cfg["model"], in_tok, cached_tok, out_tok)
                    keys = dict(experiment=experiment, condition=cfg["label"],
                                model=cfg["model"], effort=cfg["effort"], run=run, facility=name)
                    cost_row = dict(**keys, input_tokens=in_tok, cached_tokens=cached_tok,
                                    output_tokens=out_tok, cost_usd=round(call_cost, 6))
                    append_row(cost_csv, cost_row)
                    append_row(time_csv, dict(**keys,
                                              start=fmt_time(call_start), end=fmt_time(call_end),
                                              elapsed=elapsed_sec(call_start, call_end)))
                    cost_log.append(cost_row)
                    total_cost += call_cost
                    run_in     += in_tok
                    run_cached += cached_tok
                    run_out    += out_tok
                    run_cost   += call_cost
                    time.sleep(SLEEP)

                print(f"\n  [run {run} 合計]  入力={run_in} tok (うちキャッシュ={run_cached})"
                      f"  出力={run_out} tok  コスト=${run_cost:.6f}")

    finally:
        # 中断(Ctrl-C など)した場合も、今回の実行時間とサマリーを残す
        exp_end = datetime.now()
        append_row(time_csv, dict(experiment=experiment, condition="ALL",
                                  model="", effort="", run="", facility="",
                                  start=fmt_time(exp_start), end=fmt_time(exp_end),
                                  elapsed=elapsed_sec(exp_start, exp_end)))

        # ================================================================
        # 6. サマリー
        # ================================================================
        n_left = sum(is_blank(df.at[i, c]) for c in planned for i in target_df.index)
        print(f"\n\n結果CSV保存: {out_csv}")
        if n_left:
            print(f"[未完了] 未記入のセルが {n_left} 個あります(エラーを含む)。"
                  f"RESUME = True のまま再実行すると続きから再開します。")

        # コストサマリー
        print("\n" + "="*60)
        print("使用コスト サマリー(今回の実行分)")
        print("="*60)
        if cost_log:
            print(pd.DataFrame(cost_log).to_string(index=False))
        print(f"\n{'合計コスト':>20} : ${total_cost:.6f}  (約 ${total_cost:.4f})")
        print(f"コストCSV保存: {cost_csv}")

        # 実行時間サマリー
        print("\n" + "="*60)
        print("実行時間 サマリー(今回の実行分)")
        print("="*60)
        print(f"{'開始時刻':>20} : {fmt_time(exp_start)}")
        print(f"{'終了時刻':>20} : {fmt_time(exp_end)}")
        print(f"{'実行時間':>20} : {elapsed_sec(exp_start, exp_end)} 秒")
        print(f"時間CSV保存: {time_csv}")


# ================================================================
# 7. エントリーポイント
# ================================================================
if __name__ == "__main__":
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY が設定されていません。\n"
            ".env ファイルに OPENAI_API_KEY=sk-... を記載してください。"
        )
    print(f"条件      : {[c['label'] for c in CONDITIONS]}")
    print(f"対象施設  : {TARGET_Facility}  ({len(TARGET_Facility)} 件)")
    print(f"繰り返し  : {N_RUNS} 回\n")
    run_experiment()
