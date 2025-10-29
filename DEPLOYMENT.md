# v5JL Deployment Documentation

## Production URL
**Live site:** https://www.slimpunt.nl/v5JL/

## Server Architecture

### Services Overview
All services run on Digital Ocean VPS managed by PM2:

| Service | PM2 Name | Port | Description |
|---------|----------|------|-------------|
| Frontend | v5jl-chat | 5011 | Flask server serving HTML/JS/CSS |
| Backend API | v5jl-backend | 5010 | Flask API with OpenAI integration |
| JupyterLab | v5jl-jupyterlab | 8888 | Jupyter notebook server |

### Nginx Configuration

#### Frontend Route
```nginx
location /v5JL {
    # Proxy to Flask chat server
    proxy_pass http://127.0.0.1:5011/;
    # ... proxy headers ...
}
```

#### Backend API Route
```nginx
location /v5JL/api/ {
    auth_basic off;  # Bypass authentication for API
    # Proxy to Flask backend (trailing slash strips /v5JL/api prefix)
    proxy_pass http://127.0.0.1:5010/;
    # ... proxy headers ...
}
```

**Important**: The trailing slashes in both `location` and `proxy_pass` are critical:
- `location /v5JL/api/` + `proxy_pass http://127.0.0.1:5010/` → strips `/v5JL/api` prefix
- Request: `https://slimpunt.nl/v5JL/api/health` → proxies to `http://127.0.0.1:5010/health`

## Deployment Process

### 1. Local Development
```bash
# Backend
cd "/home/slimpunt/0-BRON/0-Nieuwe Projecten/projecten/v5JL"
poetry run python backend/app.py

# Frontend
poetry run python chat/server.py
```

### 2. Deploy to Production
```bash
# Sync files to VPS
rsync -avz \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='*.joblib' \
  --exclude='poetry.lock' \
  --exclude='.venv' \
  --exclude='venv' \
  -e ssh \
  "/home/slimpunt/0-BRON/0-Nieuwe Projecten/projecten/v5JL/" \
  slimpunt-vps:/var/www/slimpunt.nl/v5JL/

# Restart services
ssh slimpunt-vps "cd /var/www/slimpunt.nl/v5JL && pm2 restart v5jl-backend"
ssh slimpunt-vps "cd /var/www/slimpunt.nl/v5JL && pm2 restart v5jl-chat"
```

### 3. Verify Deployment
```bash
# Check PM2 status
ssh slimpunt-vps "pm2 list | grep v5"

# Check logs
ssh slimpunt-vps "pm2 logs v5jl-backend --lines 50"
ssh slimpunt-vps "pm2 logs v5jl-chat --lines 50"
```

## Key Features

### Minimal AI Code Generation
System prompt configured for minimal, efficient code:
- Only generates requested code (3-5 lines for simple tasks)
- NO automatic statistics or extra features
- NO educational comments unless requested
- Smart decision logic: simple imports by default, fuzzy loading on failure

### Frontend Features
- **Responsive file indicator**: Adapts to mobile/desktop screens
- **Direct-send suggestion buttons**: English prompts that execute immediately
- **Context-aware suggestions**: Based on output type (data quality, visualizations, etc.)
- **No display() errors**: Uses `print()` + `.to_string()` for all DataFrame output

### Backend Configuration
- **Port**: 5010 (production), 5000 (development)
- **OpenAI Integration**: GPT-4 for code generation
- **Minimal code policy**: Enforced via system prompt
- **Fuzzy file loading**: Smart encoding detection with chardet

## Recent Fixes (October 2025)

### 1. Display() Error Fix
**Problem**: AI generated `display()` calls causing AttributeError
**Solution**: Updated system prompt to NEVER use display(), always use `print()` with `.to_string()`

### 2. Code Over-Generation
**Problem**: 136 lines generated for simple CSV import
**Solution**: Implemented smart decision logic - 3-5 lines by default, advanced features only on request

### 3. Suggestion Button Behavior
**Problem**: Buttons filled input instead of sending directly
**Solution**: Changed onclick handler to call `sendMessage()` immediately

### 4. File Indicator Position
**Problem**: Overlapped buttons on mobile screens
**Solution**: Made responsive with media queries (left: 10px mobile, centered desktop)

### 5. Deployment Issues
**Problem**: Old interface showing after deployment
**Solutions**:
- Created missing `chat/start_production.py`
- Started PM2 `v5jl-chat` process
- Updated nginx to proxy Flask servers instead of serving static files
- Fixed frontend API URL from `http://localhost:5000` to `/v5JL/api`

### 6. Production API 404 Errors (October 22, 2025)
**Problem**: API endpoint returning 404 errors in production
**Root Causes**:
1. Old ainotebook process occupying port 5000
2. Nginx proxying to wrong port
3. Nginx not stripping `/v5JL/api` prefix from requests
4. Basic authentication blocking API requests

**Solutions**:
1. Killed orphaned ainotebook process (PID 1231002)
2. Updated nginx to proxy to correct port (5010)
3. Added trailing slashes: `location /v5JL/api/` + `proxy_pass http://127.0.0.1:5010/`
4. Added `auth_basic off;` to API location block
5. Updated frontend auto-detection for localhost vs production

**Verification**: `curl https://www.slimpunt.nl/v5JL/api/health` returns `{"message":"Backend is running","status":"ok"}`

## File Structure

```
v5JL/
├── backend/
│   ├── app.py                 # Main backend API
│   └── start_production.py    # Production wrapper (port 5010)
├── chat/
│   ├── v4style.html          # Frontend HTML
│   ├── v4style.js            # Frontend JavaScript
│   ├── server.py             # Development server (port 5001)
│   └── start_production.py   # Production wrapper (port 5011)
├── notebooks/                # Jupyter notebook storage
├── uploads/                  # File upload directory
├── ecosystem.json           # PM2 configuration
├── pyproject.toml           # Poetry dependencies
└── DEPLOYMENT.md           # This file
```

## Environment Variables

Required in `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## Troubleshooting

### Backend Not Responding
```bash
# Check if backend is running
ssh slimpunt-vps "pm2 list | grep v5jl-backend"

# Check backend logs
ssh slimpunt-vps "pm2 logs v5jl-backend"

# Restart backend
ssh slimpunt-vps "pm2 restart v5jl-backend"
```

### Frontend Not Loading
```bash
# Check if chat server is running
ssh slimpunt-vps "pm2 list | grep v5jl-chat"

# Check nginx configuration
ssh slimpunt-vps "sudo nginx -t"

# Reload nginx
ssh slimpunt-vps "sudo systemctl reload nginx"
```

### Port Issues
```bash
# Check what's listening on ports
ssh slimpunt-vps "sudo netstat -tulpn | grep ':5010\|:5011'"
```

## Git Repository

### Commit and Push
```bash
cd "/home/slimpunt/0-BRON/0-Nieuwe Projecten/projecten/v5JL"

# Add all changes
git add .

# Commit with message
git commit -m "Your commit message

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to GitHub
git push origin master
```

## Maintenance

### Update Dependencies
```bash
# Update pyproject.toml
poetry add package_name@latest

# Lock dependencies
poetry lock

# Deploy to production
rsync -avz pyproject.toml poetry.lock slimpunt-vps:/var/www/slimpunt.nl/v5JL/

# Install on VPS
ssh slimpunt-vps "cd /var/www/slimpunt.nl/v5JL && poetry install"

# Restart services
ssh slimpunt-vps "pm2 restart v5jl-backend v5jl-chat"
```

### Monitor Logs
```bash
# Real-time monitoring
ssh slimpunt-vps "pm2 logs v5jl-backend v5jl-chat"

# Check nginx access logs
ssh slimpunt-vps "sudo tail -f /var/log/nginx/access.log | grep v5JL"

# Check nginx error logs
ssh slimpunt-vps "sudo tail -f /var/log/nginx/error.log"
```

## Performance Optimization

### Current Settings
- **Minimal code generation**: Reduces token usage and response time
- **No automatic reports**: Only generate when explicitly requested
- **Direct proxy**: Nginx → Flask (no static file serving overhead)
- **PM2 auto-restart**: Ensures high availability

### Monitoring
- Use PM2 to monitor memory usage: `pm2 monit`
- Check backend response times in browser DevTools
- Monitor OpenAI API usage in OpenAI dashboard

## Support

For issues or questions:
1. Check PM2 logs: `pm2 logs v5jl-backend v5jl-chat`
2. Check nginx logs: `/var/log/nginx/error.log`
3. Verify all services running: `pm2 list`
4. Test backend directly: `curl http://localhost:5010/health`
5. Test frontend directly: `curl http://localhost:5011/`
