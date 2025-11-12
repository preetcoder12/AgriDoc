import os
import sys
import torch
import pandas as pd
import numpy as np
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF

# Try importing CNN module properly
try:
    import CNN
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        import CNN
    except ModuleNotFoundError:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            from AgriDoc import CNN
        except ModuleNotFoundError:
            print("ERROR: Could not import CNN module. Please check file structure.")
            class CNN:
                def __init__(self, num_classes):
                    self.num_classes = num_classes
                def __call__(self, x):
                    return np.zeros((1, self.num_classes))
                def eval(self):
                    pass
                def load_state_dict(self, state_dict):
                    pass

# Define directory for storing model file
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, 'datasetofdisease.pt')

# Define alternative download method for Render deployment
def download_model_directly():
    try:
        import requests
        import re
        from urllib.parse import unquote
        
        print(f"Attempting direct download of model file to {model_path}")
        
        # Your Google Drive file ID
        file_id = '1En73N317hTlvJpZDa-FqsMsIMskzU70h'
        
        # First, get the download token
        URL = f"https://drive.google.com/uc?id={file_id}&export=download"
        session = requests.Session()
        
        print(f"Getting Google Drive download token from {URL}")
        response = session.get(URL, stream=True)
        
        # Try to find the confirm token in the response
        token = None
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
                
        if token:
            print(f"Found token: {token}")
            params = {'id': file_id, 'confirm': token, 'export': 'download'}
            response = session.get("https://drive.google.com/uc", params=params, stream=True)
        else:
            print("No token needed, attempting direct download")
            
        # Save the file
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Model downloaded successfully to {model_path}")
            return True
        else:
            print(f"Failed to download model: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

# Try to load the model
def load_model_safely():
    # Check if model file exists at the expected location
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        
        # Try direct download method first (more reliable on Render)
        if download_model_directly():
            print("Direct download successful")
        else:
            # Fall back to gdown if direct download fails
            try:
                import gdown
                file_id = '1En73N317hTlvJpZDa-FqsMsIMskzU70h'
                url = f'https://drive.google.com/uc?id={file_id}'
                print(f"Attempting to download with gdown from {url}")
                gdown.download(url, model_path, quiet=False)
            except Exception as e:
                print(f"gdown download failed: {str(e)}")
                print("WARNING: Could not download model. App will run with limited functionality.")
                return CNN.CNN(39)  # Return dummy model
    
    try:
        # Load model using absolute path
        print(f"Loading model from {model_path}")
        model = CNN.CNN(39)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("WARNING: Running with limited functionality")
        # Return a dummy model that will return zeros
        return CNN.CNN(39)

# Load disease and supplement info with error handling
def load_data_safely():
    try:
        # Use absolute paths for CSV files
        base_dir = os.path.dirname(os.path.abspath(__file__))
        disease_csv = os.path.join(base_dir, 'disease_info.csv')
        supplement_csv = os.path.join(base_dir, 'supplement_info.csv')
        
        disease_info = pd.read_csv(disease_csv, encoding='cp1252')
        supplement_info = pd.read_csv(supplement_csv, encoding='cp1252')
        return disease_info, supplement_info
    except Exception as e:
        print(f"Error loading CSV data: {str(e)}")
        # Return empty DataFrames with expected columns as fallback
        disease_cols = ['disease_name', 'description', 'Possible Steps', 'image_url']
        supplement_cols = ['supplement name', 'supplement image', 'buy link']
        
        return pd.DataFrame(columns=disease_cols), pd.DataFrame(columns=supplement_cols)

# Load model and data
model = load_model_safely()
disease_info, supplement_info = load_data_safely()

# Prediction function with error handling
def prediction(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image)
        input_data = input_data.view((-1, 3, 224, 224))
        output = model(input_data)
        
        # Handle both tensor and numpy outputs
        if isinstance(output, torch.Tensor):
            output = output.detach().numpy()
        
        index = np.argmax(output)
        return index
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return 0  # Return default index in case of error

# Flask app
app = Flask(__name__)

# Ensure uploads directory exists
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        try:
            image = request.files['image']
            filename = image.filename
            file_path = os.path.join(UPLOAD_DIR, filename)
            image.save(file_path)
            
            pred = prediction(file_path)
            
            # Check if pred is within valid range
            if pred >= len(disease_info):
                pred = 0  # Use first disease as fallback
                
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            
            # Same for supplement info
            if pred >= len(supplement_info):
                pred = 0
                
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]
            
            return render_template('submit.html', 
                                   title=title, 
                                   desc=description, 
                                   prevent=prevent,
                                   image_url=image_url, 
                                   pred=pred, 
                                   sname=supplement_name,
                                   simage=supplement_image_url, 
                                   buy_link=supplement_buy_link)
        except Exception as e:
            return render_template('error.html', error=str(e))

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']),
                           disease=list(disease_info['disease_name']), 
                           buy=list(supplement_info['buy link']))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # In a real application, you would process the form data here
        # For example, send an email or save to database
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        subject = request.form.get('subject', '')
        message = request.form.get('message', '')
        
        # You could add email sending functionality here
        # For now, we'll just render the contact page with a success message
        return render_template('contact.html', success=True, 
                             message='Thank you for your message! We will get back to you soon.')
    
    return render_template('contact.html')

# Add a simple health check endpoint
@app.route('/health')
def health_check():
    return {'status': 'ok'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)