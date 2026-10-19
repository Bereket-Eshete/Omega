import providers


def result_call(tools):
    request = providers.LLMRequest().with_tools(tools)
    response = providers.LLMResponse().add_tool_call(
        providers.LLMToolCall().with_name('ctx-add-result').with_id('frame-result-1')
        .with_arguments({'variant': 'checked', 'metrics': 'frame tool result'}))
    response = providers._validate_response(request, response)
    assert not response.is_error()
    assert not response.calls[0].is_error()
    return providers.llmResponseToSExpr(response)


def has_completion_tool(tools):
    matches = [tool for tool in tools if tool.name == 'complete-goals-stm']
    return len(matches) == 1 and [p.name for p in matches[0].parameters] == ['summary']


def has_all_frame_tools(tools):
    expected = {
        'new-frame', 'new-autonomous-frame', 'switch-mode', 'switch-frame',
        'send_probe', 'show-root-frame', 'show-current-frame', 'show-frame-index',
        'show-active-framespace', 'show-completed-framespace', 'show-frame-relation',
        'ctx-add-hypothesis', 'ctx-add-result', 'complete-goals-stm',
        'complete-goals-ltm', 'clear-frame-junk',
    }
    return expected <= {tool.name for tool in tools}
