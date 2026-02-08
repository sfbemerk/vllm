# Fix for Reasoning Parser + Speculative Decoding + Structured Output Interaction

## Problem Description

When combining reasoning parser, speculative decoding, and structured output/grammar, tokens generated after the reasoning end token are returned as content that don't satisfy the grammar. This is because grammar is only active after reasoning has completed, but the speculative tokens generated BEFORE reasoning ends are not validated against the grammar.

## Root Cause

The issue occurs in the following sequence:

1. **During reasoning phase**: Speculative tokens are generated but NOT validated against grammar (because `should_advance()` returns `False` when `reasoning_ended` is `False`)

2. **Reasoning ends**: In `update_from_output()`, `should_advance()` is called and sets `reasoning_ended = True`

3. **In `update_draft_token_ids_in_output()`**: Speculative tokens are validated against grammar, but at this point `reasoning_ended` is already `True`, so they get validated

4. **The bug**: However, these spec tokens were generated BEFORE reasoning ended, meaning they represent tokens from the reasoning phase that should NOT be accepted as the "actual" output (e.g., only the reasoning content, not the structured response). Since these tokens bypassed grammar validation during generation, they can leak into the final output.

## Fix Location

The fix is in `vllm/v1/core/sched/scheduler.py` in the `update_draft_token_ids()` method (lines 1596-1620).

## Fix Logic

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
```

## Key Points

1. **Detect transition**: We store the `reasoning_ended` state BEFORE calling `should_advance()` and compare it to the state AFTER calling it

2. **Discard unvalidated tokens**: If reasoning transitions (False → True), we discard all speculative tokens generated in this step because:
   - They were not validated against the grammar during generation
   - They likely represent reasoning content, not structured output

3. **No impact on normal flow**: If reasoning has already ended or hasn't ended yet, the behavior is unchanged

4. **Preserves performance**: We only add this check when reasoning is in use (`reasoner is not None`)

## Testing

A test has been added at:
- `tests/v1/structured_output/test_spec_decode_reasoning_interaction.py`

The key test case:
- Simulates spec tokens generated during reasoning
- Verifies they are discarded when reasoning ends
- Ensures tokens after reasoning ends ARE validated

## Related Files

- `vllm/v1/core/sched/scheduler.py` - Main fix location
- `vllm/v1/structured_output/__init__.py` - Contains `should_advance()` and `should_fill_bitmask()` methods
- `tests/v1/structured_output/test_reasoning_structured_output.py` - Existing tests for reasoning + structured output

## Example Scenario

Without fix:
```
[reasoning tokens...] [spec token 1, spec token 2, spec token 3] <- reasoning ends here
```
The spec tokens (1, 2, 3) would be validated as if they were structured output, potentially accepting invalid tokens into the response.

With fix:
```
[reasoning tokens...] [spec token 1, spec token 2, spec token 3] <- reasoning ends here
```
The spec tokens are discarded, and new spec tokens will be generated AFTER reasoning ends, which will be properly validated against the grammar.