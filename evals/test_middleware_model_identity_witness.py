"""Hermes upstream main independent middleware-only witness; isolated fake wire."""
from types import SimpleNamespace

def _response():
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="ok",tool_calls=[],reasoning=None),
        finish_reason="stop")],usage=None)

class _FakeCompletions:
    def __init__(self,sink):self.sink=sink
    def create(self,**kwargs):
        self.sink.append(kwargs)
        return _response()

class _FakeClient:
    def __init__(self,sink):
        self.chat=SimpleNamespace(completions=_FakeCompletions(sink))
    def close(self):pass

class _Manifest:
    name="main-model-identity-witness"
    key="main-model-identity-witness"
    source="bundled"

def test_main_middleware_request_model_and_observer_disagree(monkeypatch):
    from hermes_cli.plugins import get_plugin_manager,PluginContext
    from agent import relay_llm
    requests=[]; hooks=[]; relays=[]
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **kwargs:_FakeClient(requests))
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client",lambda *args,**kwargs:_FakeClient(requests))
    monkeypatch.setattr("model_tools.get_tool_definitions",lambda *a,**kw:[])
    monkeypatch.setattr("agent.agent_init._routed_client_kwargs",lambda *a,**kw:{})
    monkeypatch.setattr("agent.model_metadata.get_model_context_length",
                        lambda model,**kw:{"override-model":128000,"base-model":256000}.get(str(model),256000))
    original_relay=relay_llm.execute
    def relay_probe(req,callback,*,name,model_name,**kw):
        relays.append((req.get("model"),model_name))
        return original_relay(req,callback,name=name,model_name=model_name,**kw)
    monkeypatch.setattr(relay_llm,"execute",relay_probe)
    from run_agent import AIAgent
    agent=AIAgent(model="base-model",provider="openai",platform="cli",
                  max_iterations=3,quiet_mode=True,skip_memory=True)
    agent._disable_streaming=True
    mgr=get_plugin_manager()
    ctx=object.__new__(PluginContext)
    ctx.manifest=_Manifest()
    ctx._manager=mgr
    def rewrite(**kw):
        req=dict(kw["request"]);req["model"]="override-model"
        return {"request":req,"source":"isolated-main-proof"}
    def observe(**kw):
        hooks.append((kw.get("model"),((kw.get("request") or {}).get("body") or {}).get("model"),
                      agent.context_compressor.model,agent.context_compressor.context_length))
    h1=ctx.register_middleware("llm_request",rewrite)
    h2=ctx.register_hook("pre_api_request",observe)
    try:
        result=agent.run_conversation("hello")
        assert "ok" in (result.get("final_response") or "")
        assert requests and requests[0]["model"]=="override-model"
        assert hooks and hooks[0][0]=="base-model" and hooks[0][1]=="override-model"
        assert hooks[0][2]=="base-model"
        assert relays and relays[0]==("override-model","base-model")
        print("UPSTREAM_MAIN_WITNESS wire=B hook=A hook.request=B relay.name=A compressor=A context="+str(hooks[0][3]))
    finally:
        h1.dispose();h2.dispose()
