# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for reasoning-aware structured output functionality (PR #25515)."""

from unittest.mock import Mock

import pytest

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.reasoning import ReasoningParser
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

THINK_END_TOKEN = 99
REASONING_TOKEN_A = 10
REASONING_TOKEN_B = 11
JSON_TOKEN_A = 20
JSON_TOKEN_B = 21


class TestReasoningStructuredOutput:
    """Test reasoning-aware structured output functionality."""

    @pytest.fixture
    def mock_model_config(self):
        """Create a mock ModelConfig."""
        config = Mock(spec=ModelConfig)
        config.skip_tokenizer_init = True  # Skip tokenizer init to avoid network calls
        config.get_vocab_size = Mock(return_value=50000)
        # Add missing runner_type attribute that tokenizer initialization expects
        config.runner_type = "generate"
        # Add other attributes that tokenizer initialization might need
        config.tokenizer = "test-tokenizer"
        config.tokenizer_mode = "auto"
        config.trust_remote_code = False
        config.tokenizer_revision = None
        return config

    @pytest.fixture
    def mock_scheduler_config(self):
        """Create a mock SchedulerConfig."""
        config = Mock(spec=SchedulerConfig)
        config.max_num_seqs = 128
        return config

    @pytest.fixture
    def mock_vllm_config(self, mock_model_config, mock_scheduler_config):
        """Create a mock VllmConfig."""
        config = Mock(spec=VllmConfig)
        config.model_config = mock_model_config
        config.scheduler_config = mock_scheduler_config
        config.structured_outputs_config = Mock()
        config.structured_outputs_config.reasoning_parser = None
        config.structured_outputs_config.enable_in_reasoning = False
        config.speculative_config = None
        return config

    @pytest.fixture
    def mock_reasoning_parser(self):
        """Create a mock ReasoningParser."""
        parser = Mock(spec=ReasoningParser)
        parser.is_reasoning_end = Mock(return_value=False)
        return parser

    @pytest.fixture
    def mock_request_with_structured_output(self):
        """Create a mock request with structured output."""
        request = Mock(spec=Request)
        request.structured_output_request = Mock()
        request.structured_output_request.reasoning_ended = None
        request.structured_output_request.grammar = Mock()
        request.structured_output_request.grammar.is_terminated = Mock(
            return_value=False
        )
        request.use_structured_output = True
        request.prompt_token_ids = [1, 2, 3, 4, 5]
        request.all_token_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        request.num_computed_tokens = 5
        request.num_output_placeholders = 0
        return request

    def test_should_fill_bitmask_with_enable_in_reasoning(
        self, mock_vllm_config, mock_request_with_structured_output
    ):
        """Test should_fill_bitmask when enable_in_reasoning is True."""
        # Enable enable_in_reasoning
        mock_vllm_config.structured_outputs_config.enable_in_reasoning = True

        manager = StructuredOutputManager(mock_vllm_config)

        # Should always return True when enable_in_reasoning is enabled
        result = manager.should_fill_bitmask(mock_request_with_structured_output)
        assert result is True

    def test_should_fill_bitmask_without_enable_in_reasoning(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_fill_bitmask when enable_in_reasoning is False."""
        # Keep enable_in_reasoning as False (default)
        config = mock_vllm_config.structured_outputs_config
        assert config.enable_in_reasoning is False

        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Mock reasoning not ended
        mock_reasoning_parser.is_reasoning_end.return_value = False

        result = manager.should_fill_bitmask(mock_request_with_structured_output)

        # Should set reasoning_ended and return its value
        assert (
            mock_request_with_structured_output.structured_output_request.reasoning_ended
            is False
        )
        assert result is False

    def test_should_fill_bitmask_no_reasoner(
        self, mock_vllm_config, mock_request_with_structured_output
    ):
        """Test should_fill_bitmask when no reasoner is configured."""
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = None

        result = manager.should_fill_bitmask(mock_request_with_structured_output)

        # Should default to True when no reasoner
        assert result is True

    def test_should_advance_with_enable_in_reasoning(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_advance when enable_in_reasoning is True."""
        # Enable enable_in_reasoning
        mock_vllm_config.structured_outputs_config.enable_in_reasoning = True

        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Should always return True when enable_in_reasoning is enabled
        result = manager.should_advance(mock_request_with_structured_output)
        assert result is True

    def test_should_advance_reasoning_not_ended(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_advance when reasoning has not ended."""
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Set reasoning as not ended
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False
        mock_reasoning_parser.is_reasoning_end.return_value = False

        result = manager.should_advance(mock_request_with_structured_output)

        # Should return False since reasoning hasn't ended
        assert result is False

    def test_should_advance_reasoning_just_ended(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_advance when reasoning ends in current step."""
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Set reasoning as not ended initially, but ends in this step
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = False
        mock_reasoning_parser.is_reasoning_end.return_value = True

        result = manager.should_advance(mock_request_with_structured_output)

        # Should set reasoning_ended to True but return False for this step
        assert (
            mock_request_with_structured_output.structured_output_request.reasoning_ended
            is True
        )
        assert result is False

    def test_should_advance_reasoning_already_ended(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """Test should_advance when reasoning has already ended."""
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        # Set reasoning as already ended
        (
            mock_request_with_structured_output.structured_output_request
        ).reasoning_ended = True

        result = manager.should_advance(mock_request_with_structured_output)

        # Should return True since reasoning has ended
        assert result is True

    def test_should_advance_with_new_token_ids_reasoning_ends_mid_batch(
        self,
        mock_vllm_config,
        mock_request_with_structured_output,
        mock_reasoning_parser,
    ):
        """When speculative decoding produces tokens containing </think>,
        should_advance(new_token_ids=...) must return True so the caller
        can process the post-reasoning tokens."""
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        struct_req = mock_request_with_structured_output.structured_output_request
        struct_req.reasoning_ended = False

        new_token_ids = [REASONING_TOKEN_A, THINK_END_TOKEN, JSON_TOKEN_A, JSON_TOKEN_B]
        mock_reasoning_parser.is_reasoning_end_streaming.return_value = True

        result = manager.should_advance(
            mock_request_with_structured_output,
            new_token_ids=new_token_ids,
        )

        assert result is True
        assert struct_req.reasoning_ended is True

    def _make_manager_for_bitmask_test(
        self, mock_vllm_config, mock_reasoning_parser, num_spec_tokens=5
    ):
        """Helper: create a StructuredOutputManager wired up for bitmask
        tests with a mock backend and pre-allocated bitmask tensor."""
        import torch

        mock_vllm_config.speculative_config = Mock()
        mock_vllm_config.speculative_config.num_speculative_tokens = num_spec_tokens
        manager = StructuredOutputManager(mock_vllm_config)
        manager.reasoner = mock_reasoning_parser

        max_entries = mock_vllm_config.scheduler_config.max_num_seqs * (
            1 + num_spec_tokens
        )
        manager._grammar_bitmask = torch.zeros(max_entries, 1, dtype=torch.int32)
        manager.backend = Mock()
        return manager

    def _make_grammar_mock(self):
        """Helper: create a mock grammar that tracks calls."""
        grammar = Mock()
        grammar.is_terminated.return_value = False
        grammar.accept_tokens.return_value = True
        grammar.fill_bitmask = Mock()
        grammar.rollback = Mock()
        return grammar

    def test_grammar_bitmask_regression_think_end_in_spec_tokens(
        self,
        mock_vllm_config,
        mock_reasoning_parser,
    ):
        """Regression test for production crash: when reasoning_ended=True
        and spec tokens contain </think>, grammar must NOT try to accept
        the </think> token.

        Production scenario: previous step set reasoning_ended=True,
        should_fill_bitmask returns True, spec tokens = [</think>],
        grammar.accept_tokens(</think>) crashes because </think> is not
        valid structured output."""
        manager = self._make_manager_for_bitmask_test(
            mock_vllm_config, mock_reasoning_parser, num_spec_tokens=1
        )
        grammar = self._make_grammar_mock()

        def mock_streaming(all_ids, delta):
            return delta == [THINK_END_TOKEN]

        mock_reasoning_parser.is_reasoning_end_streaming.side_effect = mock_streaming
        mock_reasoning_parser.is_reasoning_end.return_value = True

        request = Mock(spec=Request)
        request.structured_output_request = Mock()
        request.structured_output_request.reasoning_ended = True
        request.structured_output_request.grammar = grammar
        request.use_structured_output = True
        request.prompt_token_ids = [1, 2, 3]

        req_id = "req-crash-test"

        result = manager.grammar_bitmask(
            requests={req_id: request},
            structured_output_request_ids=[req_id],
            scheduled_spec_decode_tokens={req_id: [THINK_END_TOKEN]},
        )

        # grammar.accept_tokens must NOT have been called with </think>
        for call in grammar.accept_tokens.call_args_list:
            _, tokens = call[0]
            assert THINK_END_TOKEN not in tokens, (
                f"grammar.accept_tokens was called with </think> token "
                f"{THINK_END_TOKEN}, which should be skipped"
            )
        assert result is not None

    def test_grammar_bitmask_reasoning_ends_mid_speculation(
        self,
        mock_vllm_config,
        mock_reasoning_parser,
    ):
        """When reasoning_end appears mid-speculation, only post-reasoning
        tokens should be grammar-constrained and accepted (then rolled
        back).  grammar_bitmask() must NOT permanently set reasoning_ended
        because draft tokens have not yet been verified by rejection
        sampling; reasoning_ended is only set later by update_from_output()
        once the tokens are confirmed accepted."""
        manager = self._make_manager_for_bitmask_test(
            mock_vllm_config, mock_reasoning_parser, num_spec_tokens=5
        )
        grammar = self._make_grammar_mock()

        def mock_streaming(all_ids, delta):
            return delta == [THINK_END_TOKEN]

        mock_reasoning_parser.is_reasoning_end_streaming.side_effect = mock_streaming
        mock_reasoning_parser.is_reasoning_end.return_value = False

        request = Mock(spec=Request)
        request.structured_output_request = Mock()
        request.structured_output_request.reasoning_ended = False
        request.structured_output_request.grammar = grammar
        request.use_structured_output = True
        request.prompt_token_ids = [1, 2, 3]

        req_id = "req-mid-spec"
        spec_tokens = [
            REASONING_TOKEN_A,
            THINK_END_TOKEN,
            JSON_TOKEN_A,
            JSON_TOKEN_B,
        ]

        result = manager.grammar_bitmask(
            requests={req_id: request},
            structured_output_request_ids=[req_id],
            scheduled_spec_decode_tokens={req_id: spec_tokens},
        )

        assert result is not None
        assert request.structured_output_request.reasoning_ended is False

        # Only post-reasoning tokens should have been accepted
        accepted_tokens = [
            call[0][1][0] for call in grammar.accept_tokens.call_args_list
        ]
        assert THINK_END_TOKEN not in accepted_tokens
        assert REASONING_TOKEN_A not in accepted_tokens
        assert JSON_TOKEN_A in accepted_tokens
        assert JSON_TOKEN_B in accepted_tokens

        # Grammar advancements should be rolled back
        if grammar.accept_tokens.call_count > 0:
            grammar.rollback.assert_called_once_with(grammar.accept_tokens.call_count)

    def test_grammar_bitmask_no_reasoning_end_in_spec_tokens(
        self,
        mock_vllm_config,
        mock_reasoning_parser,
    ):
        """When reasoning hasn't ended and spec tokens don't contain
        the end marker, all positions should be unconstrained."""
        manager = self._make_manager_for_bitmask_test(
            mock_vllm_config, mock_reasoning_parser, num_spec_tokens=3
        )
        grammar = self._make_grammar_mock()

        mock_reasoning_parser.is_reasoning_end_streaming.return_value = False
        mock_reasoning_parser.is_reasoning_end.return_value = False

        request = Mock(spec=Request)
        request.structured_output_request = Mock()
        request.structured_output_request.reasoning_ended = False
        request.structured_output_request.grammar = grammar
        request.use_structured_output = True
        request.prompt_token_ids = [1, 2, 3]

        req_id = "req-no-end"
        spec_tokens = [REASONING_TOKEN_A, REASONING_TOKEN_B]

        result = manager.grammar_bitmask(
            requests={req_id: request},
            structured_output_request_ids=[req_id],
            scheduled_spec_decode_tokens={req_id: spec_tokens},
        )

        assert result is not None
        grammar.accept_tokens.assert_not_called()
        grammar.fill_bitmask.assert_not_called()
        assert request.structured_output_request.reasoning_ended is False

    def test_find_reasoning_end_detects_position_mismatch(
        self,
        mock_vllm_config,
        mock_reasoning_parser,
    ):
        """Verify find_reasoning_end_in_tokens detects when </think>
        appears at different positions in draft vs accepted tokens.

        This is the condition that triggers truncation of
        unconstrained post-reasoning tokens: the target model
        resampled </think> at an earlier position than the drafter
        predicted (or the drafter didn't include it at all), so the
        bitmask rows after </think> were all-unconstrained."""
        manager = StructuredOutputManager(mock_vllm_config)

        def mock_streaming(all_ids, delta):
            return THINK_END_TOKEN in delta

        mock_reasoning_parser.is_reasoning_end_streaming.side_effect = mock_streaming
        manager.reasoner = mock_reasoning_parser

        # Case 1: </think> not in drafts, but in accepted
        draft_tokens = [REASONING_TOKEN_A, REASONING_TOKEN_B]
        accepted_tokens = [THINK_END_TOKEN, JSON_TOKEN_A, JSON_TOKEN_B]
        end_in_drafts = manager.find_reasoning_end_in_tokens(draft_tokens)
        end_in_accepted = manager.find_reasoning_end_in_tokens(accepted_tokens)
        assert end_in_drafts is None
        assert end_in_accepted == 0
        # Condition: truncation needed (drafts had no reasoning end)

        # Case 2: </think> at position 2 in drafts, position 0 in accepted
        draft_tokens = [REASONING_TOKEN_A, REASONING_TOKEN_B, THINK_END_TOKEN]
        accepted_tokens = [THINK_END_TOKEN, JSON_TOKEN_A, JSON_TOKEN_B]
        end_in_drafts = manager.find_reasoning_end_in_tokens(draft_tokens)
        end_in_accepted = manager.find_reasoning_end_in_tokens(accepted_tokens)
        assert end_in_drafts == 2
        assert end_in_accepted == 0
        assert end_in_accepted < end_in_drafts
        # Condition: truncation needed (accepted end earlier than draft)

        # Case 3: </think> at same position — no truncation needed
        draft_tokens = [REASONING_TOKEN_A, THINK_END_TOKEN, JSON_TOKEN_A]
        accepted_tokens = [REASONING_TOKEN_A, THINK_END_TOKEN, JSON_TOKEN_B]
        end_in_drafts = manager.find_reasoning_end_in_tokens(draft_tokens)
        end_in_accepted = manager.find_reasoning_end_in_tokens(accepted_tokens)
        assert end_in_drafts == 1
        assert end_in_accepted == 1
        # No truncation: end_in_accepted >= end_in_drafts

        # Case 4: </think> later in accepted than in drafts — no truncation
        draft_tokens = [THINK_END_TOKEN, JSON_TOKEN_A]
        accepted_tokens = [REASONING_TOKEN_A, THINK_END_TOKEN, JSON_TOKEN_A]
        end_in_drafts = manager.find_reasoning_end_in_tokens(draft_tokens)
        end_in_accepted = manager.find_reasoning_end_in_tokens(accepted_tokens)
        assert end_in_drafts == 0
        assert end_in_accepted == 1
        assert end_in_accepted >= end_in_drafts
        # No truncation: accepted end is at or after draft end

    def test_truncate_unconstrained_post_reasoning_tokens(
        self,
        mock_vllm_config,
        mock_reasoning_parser,
    ):
        """Test that _truncate_unconstrained_post_reasoning_tokens correctly
        truncates tokens when </think> appears unexpectedly in accepted
        tokens.

        Scenario: EAGLE drafts [tok_A, tok_B, tok_C] (no </think>), but
        the target model resamples and produces [</think>, json1, json2].
        The bitmask was all-unconstrained, so json1 and json2 must be
        discarded."""
        from vllm.v1.core.sched.scheduler import Scheduler

        manager = StructuredOutputManager(mock_vllm_config)

        def mock_streaming(all_ids, delta):
            return THINK_END_TOKEN in delta

        mock_reasoning_parser.is_reasoning_end_streaming.side_effect = mock_streaming
        manager.reasoner = mock_reasoning_parser

        # Build a minimal mock scheduler with the structured output manager
        mock_scheduler = Mock(spec=Scheduler)
        mock_scheduler.structured_output_manager = manager
        # Bind the real method to the mock
        mock_scheduler._truncate_unconstrained_post_reasoning_tokens = (
            Scheduler._truncate_unconstrained_post_reasoning_tokens.__get__(
                mock_scheduler
            )
        )

        request = Mock(spec=Request)
        request.request_id = "req-test-truncate"
        request.num_computed_tokens = 10
        request.num_output_placeholders = 5

        # Case 1: </think> not in drafts, at position 0 in accepted
        # → truncate to just [THINK_END_TOKEN]
        draft = [REASONING_TOKEN_A, REASONING_TOKEN_B]
        accepted = [THINK_END_TOKEN, JSON_TOKEN_A, JSON_TOKEN_B]
        result = mock_scheduler._truncate_unconstrained_post_reasoning_tokens(
            accepted, draft, request
        )
        assert result == [THINK_END_TOKEN]
        assert request.num_computed_tokens == 8  # 10 - 2 discarded
        assert request.num_output_placeholders == 3  # 5 - 2 discarded

        # Reset counters
        request.num_computed_tokens = 10
        request.num_output_placeholders = 5

        # Case 2: </think> at position 2 in drafts, position 0 in accepted
        # → truncate to just [THINK_END_TOKEN]
        draft = [REASONING_TOKEN_A, REASONING_TOKEN_B, THINK_END_TOKEN]
        accepted = [THINK_END_TOKEN, JSON_TOKEN_A, JSON_TOKEN_B]
        result = mock_scheduler._truncate_unconstrained_post_reasoning_tokens(
            accepted, draft, request
        )
        assert result == [THINK_END_TOKEN]
        assert request.num_computed_tokens == 8
        assert request.num_output_placeholders == 3

        # Reset counters
        request.num_computed_tokens = 10
        request.num_output_placeholders = 5

        # Case 3: </think> at same position in both → no truncation
        draft = [REASONING_TOKEN_A, THINK_END_TOKEN, JSON_TOKEN_A]
        accepted = [REASONING_TOKEN_A, THINK_END_TOKEN, JSON_TOKEN_B]
        result = mock_scheduler._truncate_unconstrained_post_reasoning_tokens(
            accepted, draft, request
        )
        assert result == accepted  # unchanged
        assert request.num_computed_tokens == 10  # unchanged
        assert request.num_output_placeholders == 5  # unchanged

        # Case 4: no </think> in accepted → no truncation
        draft = [REASONING_TOKEN_A, REASONING_TOKEN_B]
        accepted = [REASONING_TOKEN_A, REASONING_TOKEN_B, JSON_TOKEN_A]
        result = mock_scheduler._truncate_unconstrained_post_reasoning_tokens(
            accepted, draft, request
        )
        assert result == accepted  # unchanged

        # Case 5: </think> is the last accepted token → no actual
        # discarding needed (nothing after it)
        request.num_computed_tokens = 10
        request.num_output_placeholders = 5
        draft = [REASONING_TOKEN_A, REASONING_TOKEN_B]
        accepted = [REASONING_TOKEN_A, THINK_END_TOKEN]
        result = mock_scheduler._truncate_unconstrained_post_reasoning_tokens(
            accepted, draft, request
        )
        assert result == [REASONING_TOKEN_A, THINK_END_TOKEN]
        assert request.num_computed_tokens == 10  # no change, nothing discarded
        assert request.num_output_placeholders == 5
