# Experiment-202611
I've annotated the concept labels such as 7 Yokohama sightseeing facilities with JTB reference. The data, program and artifacts will be published on this temporarily project.
On the Annotation the concept labels, each steps are listed below. Here is the workflow.
1. Annotate the label with editoralSummary and Guideline references(If necessary). I've used OpenAI API and Selected model was GPT-5.6. Each conditions are listed below.
   1. Model class: Terra/Sol
   2. Guideline presence or absence: S/G
   3. Reasoning effort: Low/High
   4. Runtime: 10 times for each conditions.
2. In each trials, each columns in the CSV result files are noted as below.
   gpt-5.6_{Model}_{Reasoning}_{Guideline P/A}_run{time}
   Ex. gpt-5.6_terra_low_S_run1


