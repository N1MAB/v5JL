# JupyterLab Configuration for AI Notebook v5
import os

c = get_config()  # noqa

# Server configuration
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = True
c.ServerApp.root_dir = os.path.join(os.path.dirname(__file__), 'notebooks')
c.ServerApp.allow_origin = '*'
c.ServerApp.disable_check_xsrf = False

# Allow extensions
c.ServerApp.allow_remote_access = True
c.LabApp.collaborative = False

# Notebook directory
c.ServerApp.notebook_dir = os.path.join(os.path.dirname(__file__), 'notebooks')

# Token authentication (empty for local development)
c.ServerApp.token = ''
c.ServerApp.password = ''

# File uploads
c.ServerApp.tornado_settings = {
    'headers': {
        'Content-Security-Policy': "frame-ancestors 'self' http://localhost:*"
    }
}

print("==================================================")
print("  AI Jupyter Notebook v5 - JupyterLab Edition")
print("==================================================")
print("  JupyterLab: http://localhost:8888")
print("  Backend API: http://localhost:5000")
print("  Notebooks: ./notebooks/")
print("==================================================")
