import base64
import http.client
import json
import threading
import unittest
from http.server import HTTPServer

from audio_engine import signal_fixture
from server import Handler, PORT, REPORTS, remember_report


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, method, path, body=None, host=None, origin=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        headers = {"Host": host or f"127.0.0.1:{PORT}"}
        if origin:
            headers["Origin"] = origin
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        status, payload = response.status, response.read()
        connection.close()
        return status, payload

    def test_fixture(self):
        status, payload = self.request("GET", "/api/fixture")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(payload)["synthetic"])

    def test_wrong_host(self):
        self.assertEqual(self.request("GET", "/", host="attacker.example")[0], 403)

    def test_cross_origin(self):
        self.assertEqual(self.request("POST", "/api/compare", "{}", origin="https://example.com")[0], 403)

    def test_real_upload_api(self):
        a, b = signal_fixture()
        body = json.dumps({"baseline": base64.b64encode(a).decode(), "practice": base64.b64encode(b).decode()})
        status, payload = self.request("POST", "/api/compare", body, origin=f"http://127.0.0.1:{PORT}")
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(payload)["events"]), 2)

    def test_invalid_base64(self):
        body = json.dumps({"baseline": "!", "practice": "!"})
        self.assertEqual(self.request("POST", "/api/compare", body, origin=f"http://127.0.0.1:{PORT}")[0], 400)

    def test_no_filesystem_traversal(self):
        self.assertEqual(self.request("GET", "/../server.py")[0], 404)

    def test_invalid_json_shape(self):
        self.assertEqual(self.request("POST", "/api/compare", "[]", origin=f"http://127.0.0.1:{PORT}")[0], 400)

    def test_invalid_mfcc_flag(self):
        a, b = signal_fixture()
        body = json.dumps({'baseline': base64.b64encode(a).decode(), 'practice': base64.b64encode(b).decode(), 'includeMfcc': 'yes'})
        self.assertEqual(self.request('POST', '/api/compare', body, origin=f'http://127.0.0.1:{PORT}')[0], 400)

    def test_mfcc_upload_api(self):
        a, b = signal_fixture()
        body = json.dumps({'baseline': base64.b64encode(a).decode(), 'practice': base64.b64encode(b).decode(), 'includeMfcc': True})
        status, payload = self.request('POST', '/api/compare', body, origin=f'http://127.0.0.1:{PORT}')
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(payload)['mfcc']['frames']), 80)

    def test_invalid_transcript(self):
        a, b = signal_fixture()
        body = json.dumps({'baseline': base64.b64encode(a).decode(), 'practice': base64.b64encode(b).decode(), 'transcript': 7})
        self.assertEqual(self.request('POST', '/api/compare', body, origin=f'http://127.0.0.1:{PORT}')[0], 400)

    def test_static(self):
        status, payload = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"CadenceLab", payload)

    def test_report_download(self):
        status, payload = self.request('GET', '/api/fixture')
        self.assertEqual(status, 200)
        result = json.loads(payload)
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        connection.request('GET', result['downloadUrl'], headers={'Host': f'127.0.0.1:{PORT}'})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader('Content-Disposition'), 'attachment; filename="cadencelab-report.json"')
        downloaded = json.loads(response.read())
        self.assertEqual(downloaded['events'], result['events'])
        self.assertNotIn('downloadUrl', downloaded)
        connection.close()

    def test_unknown_report(self):
        self.assertEqual(self.request('GET', '/api/reports/unknown')[0], 404)

    def test_report_cache_bounded(self):
        first = remember_report({'example': 0})
        for index in range(9):
            remember_report({'example': index})
        self.assertLessEqual(len(REPORTS), 8)
        self.assertEqual(self.request('GET', first)[0], 404)


if __name__ == "__main__":
    unittest.main()
