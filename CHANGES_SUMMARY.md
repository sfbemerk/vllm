Summary
=======

This document summarizes the fix for the bug where combining reasoning parser, speculative decoding, and structured output/grammar causes tokens after the reasoning end token to be returned as content that don't satisfy the grammar.

Files Changed
=============

1. vllm/v1/core/sched/scheduler.py (lines 1596-1620)
   - Modified update_draft_token_ids() method to detect reasoning transitions
   - Discards speculative tokens when reasoning transitions from not-ended to ended

2. tests/v1/structured_output/test_spec_decode_reasoning_interaction.py
   - Added test case for the bug

3. FIX_REASONING_SPEC_DECODE.md
   - Detailed documentation of the issue and fix

The Fix
=======

Location: vllm/v1/core/sched/scheduler.py, update_draft_token_ids() method

Before:
```python
# Add newly generated spec token ids to the request.
if self.structured_output_manager.should_advance(request):
    metadata = request.structured_output_request
    spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
request.spec_token_ids = spec_token_ids
```

After:
```python
# Check if reasoning ends in this step BEFORE validating/speculative decoding
reasoning_ended_before_update = (
    request.structured_output_request.reasoning_ended
    if request.structured_output_request
    else False
)
should_advance = self.structured_output_manager.should_advance(request)
reasoning_ended_after_update = (
    request.structured_output_request.reasoning_ended
    if request.structured_output_request
    else False
)

# If reasoning transitioned from not-ended to ended in this step,
# discard speculative tokens generated during reasoning phase
if reasoning_ended_before_update is False and reasoning_ended_after_update:
    spec_token_ids = []
elif should_advance:
    metadata = request.structured_output_request
    spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
else:
    # Still in reasoning phase, don't validate against grammar
    pass

request.spec_token_ids = spec_token_ids
```

How It Works
============

1. The fix captures the reasoning state before and after the validation step
2. If reasoning transitions (False → True) during this step, it means:
   - Speculative tokens were generated when reasoning was active
   - These tokens were NOT validated against the grammar during generation
   - They represent reasoning content, not structured output
3. Therefore, we discard these spec tokens to prevent unvalidated content from leaking
4. In subsequent steps (after reasoning has already ended), new spec tokens will be properly validated

Impact
======

- Fixes the bug where unvalidated tokens leak into the output
- No performance impact (only adds two boolean checks when reasoning is used)
- Preserves existing behavior for non-reasoning cases
- Safe for all scenarios (doesn't change logic unless reasoning is transitioning)

Testing
=======

- Added test in tests/v1/structured_output/test_spec_decode_reasoning_interaction.py
- Test simulates the exact scenario: spec tokens generated during reasoning transition
- Verifies that spec tokens are discarded when reasoning ends