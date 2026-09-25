# UrbanBind Code

This repository contains the implementation of UrbanBind.

## QA_design

Code for urban-dynamics data preparation and QA construction, including chronological data splitting, spatial rendering, QA template generation, numerical and dense-map target construction, sampling, and validation utilities.

## stage1

Code for QA-Guided Spatial-Temporal Knowledge Grounding, including Qwen2.5-VL integration, LoRA-based adaptation, auxiliary value and map prediction heads, normalization, training and evaluation, visual-grounding analysis, and representation probing.

## stage2

Code for Multi-City Urban Dynamics Encoding and Prediction, including native-grid spatial-temporal encoding, cross-city interaction, VLM-conditioned multi-projector fusion, frozen-VLM contextualization, city-specific forecasting decoders, Nash-balanced multi-city optimization, evaluation, ablation studies, and cross-city transfer utilities.

## Project Paths

Data, QA outputs, model checkpoints, caches, and experiment outputs are configured through project-relative paths or optional environment variables.