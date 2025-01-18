from aiohttp import web


class TestServer:
    def __init__(self, host='127.0.0.1', port=8765):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.app.router.add_get('/', self.handle_root)
        self.app.router.add_get('/page1', self.handle_page1)
        self.app.router.add_get('/page2', self.handle_page2)
        self.app.router.add_get('/error', self.handle_error)
        self.runner = None
        self.site = None

    async def start(self):
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        return f"http://{self.host}:{self.port}"

    async def stop(self):
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()

    async def handle_root(self, request):
        html = """
            <!DOCTYPE html>
            <html>
                <head>
                    <title>Test Documentation</title>
                    <meta charset="utf-8">
                </head>
                <body>
                    <h1>Test Documentation Index</h1>
                    <p>Welcome to the test documentation site.</p>
                    <ul>
                        <li><a href="/page1">Page 1</a></li>
                        <li><a href="/page2">Page 2</a></li>
                        <li><a href="/error">Error Page</a></li>
                    </ul>
                </body>
            </html>
        """
        return web.Response(text=html.strip(), content_type='text/html', charset='utf-8')

    async def handle_page1(self, request):
        html = """
            <!DOCTYPE html>
            <html>
                <head>
                    <title>Page 1</title>
                    <meta charset="utf-8">
                </head>
                <body>
                    <h1>Page 1: Feature Documentation</h1>
                    <p>This page contains detailed feature documentation.</p>
                    <div class="feature">
                        <h2>Feature 1</h2>
                        <p>Feature 1 is a core component that handles data processing.</p>
                        <code>example_function(data)</code>
                    </div>
                    <div class="feature">
                        <h2>Feature 2</h2>
                        <p>Feature 2 provides advanced analytics capabilities.</p>
                        <code>analyze_data(dataset)</code>
                    </div>
                    <a href="/">Back to Index</a>
                    <a href="/page2">Next: Page 2</a>
                </body>
            </html>
        """
        return web.Response(text=html.strip(), content_type='text/html', charset='utf-8')

    async def handle_page2(self, request):
        html = """
            <!DOCTYPE html>
            <html>
                <head>
                    <title>Page 2</title>
                    <meta charset="utf-8">
                </head>
                <body>
                    <h1>Page 2: API Reference</h1>
                    <p>Complete API reference documentation.</p>
                    <div class="api">
                        <h2>API Endpoints</h2>
                        <ul>
                            <li>/api/v1/data - Get data</li>
                            <li>/api/v1/analyze - Analyze data</li>
                        </ul>
                    </div>
                    <div class="examples">
                        <h2>Example Usage</h2>
                        <pre>
                            curl -X GET /api/v1/data
                            curl -X POST /api/v1/analyze
                        </pre>
                    </div>
                    <a href="/">Back to Index</a>
                    <a href="/page1">Previous: Page 1</a>
                </body>
            </html>
        """
        return web.Response(text=html.strip(), content_type='text/html', charset='utf-8')

    async def handle_error(self, request):
        raise web.HTTPInternalServerError(text="Simulated server error")
