import gevent
import gevent.pywsgi
import gevent.queue
import tinyrpc
import tinyrpc.transports
from loguru import logger
from tinyrpc.dispatch import public
from tinyrpc.protocols.msgpackrpc import MSGPACKRPCProtocol
from tinyrpc.server.gevent import RPCServerGreenlets
from tinyrpc.transports.wsgi import WsgiServerTransport

from cloud_track.foundation_model_wrappers import WrapperBase

from .utils import deserialize_args, deserialize_kwargs, serialize_results


class FmBackend:
    def __init__(self, model: WrapperBase):
        self.model = model
        self.dispatcher = tinyrpc.dispatch.RPCDispatcher()
        self.transport = WsgiServerTransport(
            queue_class=gevent.queue.Queue,
            max_content_length=99999999,
            allow_origin="*",
        )

        self.dispatcher.register_instance(self)

    @public
    def run_inference(self, *args, **kwargs):
        # deserialize the arguments
        args = deserialize_args(args)
        kwargs = deserialize_kwargs(kwargs)

        results = self.model.run_inference(*args, **kwargs)

        # serialize the results
        results = serialize_results(results)

        return results

    @public
    def run_vlm_only_batch(self, *args, **kwargs):
        # 反序列化参数
        args = deserialize_args(args)
        kwargs = deserialize_kwargs(kwargs)
        logger.info(f"run_vlm_only_batch received args: {len(args)}, kwargs: {kwargs.keys()}")
        if args:
            logger.info(f"First arg type: {type(args[0])}")
        # 调用模型方法
        return self.model.run_vlm_only_batch(*args, **kwargs)
        
    def start(self, ip: str, port: int):
        wsgi_server = gevent.pywsgi.WSGIServer(
            (ip, port), self.transport.handle
        )
        gevent.spawn(wsgi_server.serve_forever)
        rpc_server = RPCServerGreenlets(
            self.transport, MSGPACKRPCProtocol(), self.dispatcher
        )
        logger.info(f"RPC Server started at {ip}:{port}")
        logger.info(f"Model: {self.model.get_name()}")
        rpc_server.serve_forever()

    def get_model_name(self):
        return self.model.get_name()