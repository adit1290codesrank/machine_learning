from flask import Flask, request, jsonify, render_template_string
import subprocess
import threading
import time
import numpy as np
from PIL import Image

app = Flask(__name__)

process = subprocess.Popen(
    ['server.exe'], 
    stdin=subprocess.PIPE, 
    stdout=subprocess.PIPE, 
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

time.sleep(1) 
if process.poll() is None:
    startup_msg = process.stdout.readline()
    print(f"Status: {startup_msg.strip()}")
else:
    print("Error: Server failed to start.")

def log_reader():
    for line in iter(process.stderr.readline, ''):
        print(f"LOG: {line.strip()}")
threading.Thread(target=log_reader, daemon=True).start()

def preprocess_image(raw_pixels):
    arr = np.array(raw_pixels).reshape(28, 28)
    
    rows = np.any(arr > 0.1, axis=1)
    cols = np.any(arr > 0.1, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return raw_pixels
        
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    cropped = arr[ymin:ymax+1, xmin:xmax+1]
    
    img_uint8 = (cropped * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8, mode='L')
    
    w, h = img.size
    scale = 20.0 / max(w, h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    final_img = Image.new('L', (28, 28), 0)
    
    pad_x = (28 - new_w) // 2
    pad_y = (28 - new_h) // 2
    final_img.paste(img, (pad_x, pad_y))
    
    final_arr = np.array(final_img).T
    
    return (final_arr / 255.0).flatten().tolist()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
    <title>Interface</title>
    <style>
        body { background-color: #121212; color: white; font-family: sans-serif; text-align: center; margin: 0; touch-action: none; }
        #result-box { font-size: 80px; font-weight: bold; color: #00ffcc; min-height: 100px; margin-top: 20px; text-shadow: 0 0 20px rgba(0,255,204,0.3); }
        #canvas-container { position: relative; margin: 10px auto; width: 300px; height: 300px; border: 2px solid #333; border-radius: 10px; background: black; }
        canvas { width: 100%; height: 100%; cursor: crosshair; }
        .controls { margin: 20px; display: flex; justify-content: center; align-items: center; gap: 10px; }
        label { color: #aaa; font-size: 14px; }
        .btn-container { display: flex; justify-content: center; gap: 20px; margin-bottom: 30px; }
        button { padding: 15px 30px; border: none; border-radius: 8px; font-size: 18px; font-weight: bold; cursor: pointer; transition: 0.2s; }
        #btn-clear { background: #333; color: white; }
        #btn-predict { background: #007bff; color: white; }
        button:active { transform: scale(0.95); }
    </style>
</head>
<body>
    <div id="result-box">?</div>
    <div id="canvas-container">
        <canvas id="mainCanvas" width="280" height="280"></canvas>
    </div>
    <div class="controls">
        <label>Brush Size:</label>
        <input type="range" id="brushSize" min="5" max="40" value="25" oninput="updateBrush()">
    </div>
    <div class="btn-container">
        <button id="btn-clear" onclick="clearCanvas()">Clear</button>
        <button id="btn-predict" onclick="predict()">Predict</button>
    </div>
    <script>
        const canvas = document.getElementById('mainCanvas');
        const ctx = canvas.getContext('2d');
        const resultBox = document.getElementById('result-box');
        const brushSlider = document.getElementById('brushSize');
        
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'white';
        ctx.lineWidth = brushSlider.value;
        
        let isDrawing = false;

        function updateBrush() { ctx.lineWidth = brushSlider.value; }

        canvas.addEventListener('mousedown', startDraw);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', endDraw);
        canvas.addEventListener('mouseout', endDraw);
        canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDraw(e.touches[0]); });
        canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e.touches[0]); });
        canvas.addEventListener('touchend', endDraw);

        function startDraw(e) {
            isDrawing = true;
            const pos = getPos(e);
            ctx.beginPath();
            ctx.moveTo(pos.x, pos.y);
        }

        function draw(e) {
            if (!isDrawing) return;
            const pos = getPos(e);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
        }

        function endDraw() { isDrawing = false; }

        function getPos(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
        }

        function clearCanvas() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            resultBox.innerText = "?";
        }

        function predict() {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(canvas, 0, 0, 28, 28);
            
            const imgData = tempCtx.getImageData(0, 0, 28, 28);
            const data = imgData.data;
            let pixels = [];
            for (let i = 0; i < data.length; i += 4) {
                let val = data[i] / 255.0;
                pixels.push(val);
            }

            resultBox.innerText = "...";
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ pixels: pixels })
            })
            .then(res => res.json())
            .then(data => { resultBox.innerText = data.result; })
            .catch(err => { console.error(err); resultBox.innerText = "Err"; });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        raw_pixels = data.get('pixels', [])
        
        processed_pixels = preprocess_image(raw_pixels)
        pixel_string = " ".join([f"{x:.3f}" for x in processed_pixels])
        
        if process.poll() is not None:
             return jsonify({'result': 'Server Died'})

        process.stdin.write(pixel_string + "\n")
        process.stdin.flush()
        
        result = process.stdout.readline().strip()
        return jsonify({'result': result})
    except Exception as e:
        print("Error:", e)
        return jsonify({'result': 'Error'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)