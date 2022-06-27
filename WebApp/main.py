from model import classify_article_category
import tornado.ioloop
import tornado.web
import json
import os

def set_global_cors_policy(request):
    request.set_header("Access-Control-Allow-Origin", "http://localhost:8080")

def global_cors_policy_decorator(pre_request_message=None):
    def decorator(function):
        def wrapper(*args, **kwargs):
            self = args[0]
            set_global_cors_policy(self)
            if pre_request_message is not None:
                print(pre_request_message)
            result = function(*args, **kwargs)
            self.finish()
            return result
        return wrapper
    return decorator

class MainHandler(tornado.web.RequestHandler):

    @global_cors_policy_decorator()
    def options(self):
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
        self.set_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.set_status(204)

    @global_cors_policy_decorator(pre_request_message="handling POST request")
    def post(self):
        try:
            data = json.loads(self.request.body)
        except json.JSONDecodeError:
            self.set_status(400)
            return self.write("invalid json format")

        if "article" not in data:
            self.set_status(400)
            return self.write("payload: missing key[article]")

        try:
            article_category = classify_article_category(data["article"])[0]
            self.set_status(200)
            self.write(json.dumps({"category": article_category}))
        except Exception as e:
            print(e)
            return self.set_status(500)


def make_app():
    return tornado.web.Application([
        (r"/classify_article_category", MainHandler),
        (r"/(.*)", tornado.web.StaticFileHandler,
         {"path": os.path.dirname(__file__), "default_filename": "index.html"})
    ])


if __name__ == "__main__":
    port = 8080
    app = make_app()
    app.listen(port)
    print(f"listening to port {port}...")
    tornado.ioloop.IOLoop.current().start()
