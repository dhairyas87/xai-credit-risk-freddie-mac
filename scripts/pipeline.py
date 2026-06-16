# scripts/pipeline.py

PIPELINE_STAGES = [


"target_generation",

"feature_engineering",

"combine_quarters",

"feature_store",

"model_training",

"evaluation"


]

def run_pipeline(


start_stage="target_generation",

end_stage="evaluation",

force_rebuild=False


):


stages = PIPELINE_STAGES

start_idx = stages.index(
    start_stage
)

end_idx = stages.index(
    end_stage
)

stages_to_run = (
    stages[start_idx:end_idx + 1]
)

print(
    f"Running: {stages_to_run}"
)

if "combine_quarters" in stages_to_run:

    combine_quarters()

if "feature_store" in stages_to_run:

    build_feature_store()

if "model_training" in stages_to_run:

    train_all_models()

if "evaluation" in stages_to_run:

    evaluate_all_models()
