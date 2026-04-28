#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs.types import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.processor.pipeline import ObservationProcessorStep
from lerobot.utils.constants import (
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_art import ARTConfig

OBS_STATE_RAW = "observation.state_raw"


@dataclass
@ProcessorStepRegistry.register(name="copy_observation_key_processor")
class CopyObservationKeyProcessorStep(ObservationProcessorStep):
    """Copy one observation key to another without modifying the original."""

    source_key: str
    target_key: str

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        if self.source_key in observation and self.target_key not in observation:
            observation[self.target_key] = observation[self.source_key]
        return observation

    def get_config(self) -> dict[str, Any]:
        return {"source_key": self.source_key, "target_key": self.target_key}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        new_features = features.copy()
        observation_features = new_features.get(PipelineFeatureType.OBSERVATION, {}).copy()
        if self.source_key in observation_features and self.target_key not in observation_features:
            observation_features[self.target_key] = observation_features[self.source_key]
        new_features[PipelineFeatureType.OBSERVATION] = observation_features
        return new_features


def make_art_pre_post_processors(
    config: ARTConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates the pre- and post-processing pipelines for the ART policy."""

    input_steps = [RenameObservationsProcessorStep(rename_map={})]
    if config.tokenize_actions and config.tokenize_delta_actions and config.robot_state_feature:
        input_steps.append(
            CopyObservationKeyProcessorStep(source_key=OBS_STATE, target_key=OBS_STATE_RAW)
        )

    input_steps += [
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]

    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
