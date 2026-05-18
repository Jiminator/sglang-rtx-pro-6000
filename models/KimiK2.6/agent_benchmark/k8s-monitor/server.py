import http.server
import socketserver
import json
import urllib.request
import os
import ssl
import re
from socketserver import ThreadingMixIn
import urllib.parse
import subprocess

class ThreadedHTTPServer(ThreadingMixIn, socketserver.TCPServer):
    pass

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == '/api/pods':
            self.handle_api()
        elif parsed_path.path == '/api/logs':
            query = urllib.parse.parse_qs(parsed_path.query)
            pod_name = query.get('pod', [None])[0]
            if pod_name:
                self.handle_logs(pod_name)
            else:
                self.send_error(400, "Pod name required")
        elif parsed_path.path == '/api/metrics':
            query = urllib.parse.parse_qs(parsed_path.query)
            target = query.get('target', ['10.10.0.5:30000'])[0]
            self.handle_metrics(target)
        elif parsed_path.path == '/api/host-metrics':
            query = urllib.parse.parse_qs(parsed_path.query)
            pod_name = query.get('pod', [None])[0]
            if pod_name:
                self.handle_host_metrics(pod_name)
            else:
                self.send_error(400, "Pod name required")
        elif parsed_path.path == '/api/nvidia-smi':
            query = urllib.parse.parse_qs(parsed_path.query)
            pod_name = query.get('pod', [None])[0]
            if pod_name:
                self.handle_nvidia_smi(pod_name)
            else:
                self.send_error(400, "Pod name required")
        elif parsed_path.path == '/api/benchmark-status':
            query = urllib.parse.parse_qs(parsed_path.query)
            pod_name = query.get('pod', [None])[0]
            if pod_name:
                self.handle_benchmark_status(pod_name)
            else:
                self.send_error(400, "Pod name required")
        elif parsed_path.path == '/api/benchmark-summary':
            query = urllib.parse.parse_qs(parsed_path.query)
            pod_name = query.get('pod', [None])[0]
            if pod_name:
                self.handle_benchmark_summary(pod_name)
            else:
                self.send_error(400, "Pod name required")
        elif parsed_path.path == '/' or parsed_path.path == '/index.html':
            self.handle_ui()
        else:
            self.send_error(404, "File not found")

    def handle_ui(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        try:
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())
        except FileNotFoundError:
            self.wfile.write(b"index.html not found")

    def handle_api(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        pods_data = self.get_pods()
        self.wfile.write(json.dumps(pods_data).encode('utf-8'))

    def handle_logs(self, pod_name):
        self.send_response(200)
        self.send_header('Content-type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.end_headers()
        
        token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
        if not os.path.exists(token_path):
            self.wfile.write(b"data: Error: Token file not found\n\n")
            return
            
        with open(token_path, 'r') as f:
            token = f.read().strip()
            
        url = f'https://kubernetes.default.svc/api/v1/namespaces/default/pods/{pod_name}/log?follow=true&tailLines=100'
        req = urllib.request.Request(url)
        req.add_header('Authorization', f'Bearer {token}')
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        try:
            with urllib.request.urlopen(req, context=ctx) as response:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    self.wfile.write(f"data: {line.decode('utf-8')}\n\n".encode('utf-8'))
                    self.wfile.flush()
        except Exception as e:
            self.wfile.write(f"data: Error: {str(e)}\n\n".encode('utf-8'))

    def handle_metrics(self, target):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        url = f'http://{target}/metrics'
        req = urllib.request.Request(url)
        
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                data = response.read().decode('utf-8')
                metrics = self.parse_metrics(data)
                self.wfile.write(json.dumps(metrics).encode('utf-8'))
        except Exception as e:
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

    def handle_host_metrics(self, pod_name):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
        if not os.path.exists(token_path):
            self.wfile.write(json.dumps({"error": "Token file not found"}).encode('utf-8'))
            return
            
        with open(token_path, 'r') as f:
            token = f.read().strip()
            
        url = f'https://kubernetes.default.svc/apis/metrics.k8s.io/v1beta1/namespaces/default/pods/{pod_name}'
        req = urllib.request.Request(url)
        req.add_header('Authorization', f'Bearer {token}')
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        try:
            with urllib.request.urlopen(req, context=ctx) as response:
                data = json.loads(response.read().decode())
                cpu_usage = data['containers'][0]['usage']['cpu']
                mem_usage = data['containers'][0]['usage']['memory']
                
                self.wfile.write(json.dumps({
                    "cpu": self.parse_cpu(cpu_usage),
                    "memory": self.parse_memory(mem_usage)
                }).encode('utf-8'))
        except Exception as e:
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

    def handle_nvidia_smi(self, pod_name):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        
        try:
            result = subprocess.run(
                ["/tmp/kubectl", "exec", pod_name, "--", "nvidia-smi"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self.wfile.write(result.stdout.encode('utf-8'))
            else:
                self.wfile.write(f"Error running nvidia-smi: {result.stderr}".encode('utf-8'))
        except Exception as e:
            self.wfile.write(f"Exception: {str(e)}".encode('utf-8'))

    def handle_benchmark_status(self, pod_name):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        try:
            with open('sequence_steps.json', 'r') as f:
                sequence_steps = json.load(f)
        except Exception as e:
            self.wfile.write(json.dumps({"error": f"Failed to load sequence_steps: {e}"}).encode('utf-8'))
            return
            
        try:
            result = subprocess.run(
                ["/tmp/kubectl", "logs", pod_name, "--tail=10000"],
                capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0:
                self.wfile.write(json.dumps({"error": f"Failed to get logs: {result.stderr}"}).encode('utf-8'))
                return
                
            logs = result.stdout
            status = self.calculate_progress(logs, sequence_steps)
            self.wfile.write(json.dumps(status).encode('utf-8'))
        except Exception as e:
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

    def handle_benchmark_summary(self, pod_name):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        try:
            # Read full logs for completed job
            result = subprocess.run(
                ["/tmp/kubectl", "logs", pod_name],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                self.wfile.write(json.dumps({"error": f"Failed to get logs: {result.stderr}"}).encode('utf-8'))
                return
                
            logs = result.stdout
            summary = self.extract_summary(logs)
            self.wfile.write(json.dumps(summary).encode('utf-8'))
        except Exception as e:
            self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))

    def extract_summary(self, logs):
        marker = "=== BENCHMARK RESULTS ==="
        idx = logs.find(marker)
        if idx == -1:
            return {"error": "Benchmark results marker not found in logs"}
            
        json_str = logs[idx + len(marker):].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse JSON summary: {e}", "raw": json_str[:500]}

    def calculate_progress(self, logs, sequence_steps):
        completed = {}
        total_chars = 0
        success_lines = 0
        
        for line in logs.split('\n'):
            if "Success: Trace" in line:
                success_lines += 1
                match = re.search(r'Trace ([a-f0-9-]+), Step (\d+), Prompt Chars: (\d+)', line)
                if match:
                    trace_id = match.group(1)
                    step = int(match.group(2))
                    chars = int(match.group(3))
                    
                    completed[trace_id] = max(completed.get(trace_id, -1), step)
                    total_chars += chars
        
        granular = []
        for trace_id, max_step in completed.items():
            total_steps = sequence_steps.get(trace_id, 0)
            if total_steps > 0:
                completed_steps = max_step + 1
                granular.append({
                    "trace_id": trace_id,
                    "current_step": completed_steps,
                    "total_steps": total_steps,
                    "percent": min(round((completed_steps / total_steps) * 100), 100)
                })
        
        return {
            "total_steps_completed": success_lines,
            "total_steps_expected": 12542,
            "active_sequences": len(completed),
            "total_sequences_expected": 256,
            "total_characters": total_chars,
            "granular_progress": sorted(granular, key=lambda x: x['percent'], reverse=True)
        }

    def parse_cpu(self, val):
        if val.endswith('n'):
            return float(val[:-1]) / 1000000000
        if val.endswith('u'):
            return float(val[:-1]) / 1000000
        if val.endswith('m'):
            return float(val[:-1]) / 1000
        return float(val)

    def parse_memory(self, val):
        if val.endswith('Ki'):
            return float(val[:-2]) * 1024
        if val.endswith('Mi'):
            return float(val[:-2]) * 1024 * 1024
        if val.endswith('Gi'):
            return float(val[:-2]) * 1024 * 1024 * 1024
        return float(val)

    def parse_metrics(self, data):
        result = {}
        patterns = {
            "kv_used": r'^sglang:kv_used_tokens(?:\{.*?\})?\s+([\d.]+)',
            "kv_available": r'^sglang:kv_available_tokens(?:\{.*?\})?\s+([\d.]+)',
            "kv_evictable": r'^sglang:kv_evictable_tokens(?:\{.*?\})?\s+([\d.]+)',
            "hit_rate": r'^sglang:cache_hit_rate(?:\{.*?\})?\s+([\d.]+)',
            "running_reqs": r'^sglang:num_running_reqs(?:\{.*?\})?\s+([\d.]+)',
            "queue_reqs": r'^sglang:num_queue_reqs(?:\{.*?\})?\s+([\d.]+)',
            "throughput": r'^sglang:gen_throughput(?:\{.*?\})?\s+([\d.]+)',
            "spec_len": r'^sglang:spec_accept_length(?:\{.*?\})?\s+([\d.]+)',
            "spec_rate": r'^sglang:spec_accept_rate(?:\{.*?\})?\s+([\d.]+)',
            "hicache_total": r'^sglang:hicache_host_total_tokens(?:\{.*?\})?\s+([\d.]+)',
            "hicache_used": r'^sglang:hicache_host_used_tokens(?:\{.*?\})?\s+([\d.]+)'
        }
        for line in data.split('\n'):
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    result[key] = float(match.group(1))
        return result

    def get_pods(self):
        token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
        if not os.path.exists(token_path):
            return {"error": "Token file not found. Are you running in-cluster?"}
            
        with open(token_path, 'r') as f:
            token = f.read().strip()
            
        url = 'https://kubernetes.default.svc/api/v1/namespaces/default/pods'
        req = urllib.request.Request(url)
        req.add_header('Authorization', f'Bearer {token}')
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        try:
            with urllib.request.urlopen(req, context=ctx) as response:
                data = json.loads(response.read().decode())
                return self.parse_pods(data)
        except Exception as e:
            return {"error": str(e)}

    def parse_pods(self, data):
        pods = []
        for item in data.get('items', []):
            namespace = item['metadata']['namespace']
            name = item['metadata']['name']
            node = item['spec'].get('nodeName', 'Unassigned')
            status = item['status']['phase']
            pod_ip = item['status'].get('podIP', '')
            creation_ts = item['metadata'].get('creationTimestamp', '')
            
            friendly_node = node.replace('gke-shivaji-minimax-g4-', '')
            friendly_node = re.sub(r'-[a-f0-9]{8}-[a-z0-9]{4}$', '', friendly_node)
            friendly_node = re.sub(r'^[0-9]+-', '', friendly_node)
            
            # Include completed benchmark pods
            if status == 'Running' or (status == 'Succeeded' and 'benchmark' in name):
                pods.append({
                    "namespace": namespace,
                    "name": name,
                    "nodepool": friendly_node,
                    "nodeName": node,
                    "ip": pod_ip,
                    "creationTimestamp": creation_ts,
                    "status": status
                })
        return pods

PORT = 8080
with ThreadedHTTPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()
