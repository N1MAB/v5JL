"""
AI Jupyter Notebook - Python Backend
Flask server voor het uitvoeren van Python code en AI chat
"""

from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import sys
import io
import traceback
import contextlib
import base64
import os
import ast
import re
import json
import time
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use('Agg')  # Gebruik non-interactive backend
import matplotlib.pyplot as plt

# Plotly imports voor interactive visualizations
try:
    import plotly.graph_objects as go
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available")

# Load environment variables
load_dotenv()

# Set seaborn data cache to /tmp (where we have write permissions)
os.environ['SEABORN_DATA'] = '/tmp/seaborn-data'
os.makedirs('/tmp/seaborn-data', exist_ok=True)

# Validation Configuration
VALIDATION_CONFIG = {
    'enable_ai_validation': True,  # Toggle AI validation
    'max_retries': 2,
    'validator_model': 'gpt-5-nano',
    'analyzer_model': 'gpt-5-mini',
    'validator_max_tokens': 1000,
    'analyzer_max_tokens': 2000,
    'dangerous_patterns': [
        # File operations - only block destructive ones
        'os.remove', 'os.unlink', 'shutil.rmtree', 'pathlib.unlink',
        # System operations
        'subprocess.', 'os.system', 'eval(', 'exec(',
        # Dynamic imports (can bypass security)
        '__import__', 'importlib.import_module',
        # Deprecated/unsafe
        'execfile', 'file('
        # NOTE: 'open(' removed - handled separately to allow reads but block writes
    ],
    'allowed_imports': [
        # Data analysis
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn',
        'scipy', 'plotly', 'statsmodels',
        # Data sources and APIs
        'yfinance', 'requests', 'urllib', 'json', 'csv',
        # Web scraping (read-only)
        'bs4', 'beautifulsoup4', 'lxml',
        # Date/time
        'datetime', 'dateutil', 'pytz'
    ],
    'safe_file_libraries': [
        # Libraries that safely handle file I/O internally
        'pandas', 'numpy', 'json', 'csv', 'pickle', 'joblib',
        'PIL', 'pillow', 'imageio', 'h5py',
        # Geospatial libraries
        'geopandas', 'gpd', 'shapely', 'fiona', 'rasterio'
    ]
}

app = Flask(__name__)
CORS(app)  # Enable CORS voor frontend requests

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Upload folder configuration
# Point to uploads folder in parent directory (since we're in backend/)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'py', 'ipynb'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global namespace voor code execution (om variabelen te behouden tussen executions)
execution_namespace = {}


# ============================================================================
# VALIDATION CLASSES
# ============================================================================

class CodeValidator:
    """Hybrid code validator: regelgebaseerd + AI"""

    @staticmethod
    def validate_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """Check if code is syntactically valid using AST"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error op regel {e.lineno}: {e.msg}"

    @staticmethod
    def check_dangerous_patterns(code: str) -> Tuple[bool, List[str]]:
        """Check for dangerous operations"""
        dangerous_found = []

        # Check basic dangerous patterns
        for pattern in VALIDATION_CONFIG['dangerous_patterns']:
            if pattern in code:
                dangerous_found.append(pattern)

        # Check for file operations - more nuanced approach
        if 'open(' in code:
            # Check if it's using a safe library (pandas, json, etc.)
            is_safe_library_call = any(
                lib in code for lib in VALIDATION_CONFIG['safe_file_libraries']
            )

            if not is_safe_library_call:
                # Direct open() call - check for write modes
                write_modes = ["'w'", '"w"', "'a'", '"a"', "'wb'", '"wb"', "'ab'", '"ab"']
                if any(mode in code for mode in write_modes):
                    dangerous_found.append('file_write_operation')
                # Also block 'w+' and 'a+' modes (read+write)
                if any(mode in code for mode in ["'w+'", '"w+"', "'a+'", '"a+"']):
                    dangerous_found.append('file_read_write_operation')

        # Check for network file writes (FTP, SSH, etc.)
        network_write_patterns = ['paramiko', 'ftplib.FTP', 'smtplib']
        for pattern in network_write_patterns:
            if pattern in code:
                dangerous_found.append(f'network_operation_{pattern}')

        return len(dangerous_found) == 0, dangerous_found

    @staticmethod
    def ai_safety_check(code: str) -> Dict:
        """AI-based semantic safety check using GPT-5 nano"""
        if not VALIDATION_CONFIG['enable_ai_validation']:
            return {'safe': True, 'issues': [], 'suggestion': None}

        try:
            response = openai_client.chat.completions.create(
                model=VALIDATION_CONFIG['validator_model'],
                messages=[{
                    "role": "system",
                    "content": """You are a code safety validator. Analyze Python code for:
1. Security issues (file ops, system calls, dangerous imports)
2. Potential crashes (type errors, missing variables)
3. Data analysis bad practices

Respond ONLY with JSON:
{"safe": true/false, "issues": ["issue1", "issue2"], "suggestion": "how to fix"}

Be strict but practical. Allow pandas/numpy/matplotlib operations."""
                }, {
                    "role": "user",
                    "content": f"Validate this code:\n\n{code}"
                }],
                max_completion_tokens=VALIDATION_CONFIG['validator_max_tokens']
            )

            result_text = response.choices[0].message.content.strip()

            # Try to parse JSON
            try:
                result = json.loads(result_text)
                return result
            except json.JSONDecodeError:
                # If not JSON, default to safe (rule-based checks already caught dangerous patterns)
                print(f"⚠️  AI validator returned non-JSON response, defaulting to safe")
                print(f"   Response: {result_text[:200]}")
                return {'safe': True, 'issues': [], 'suggestion': None}

        except Exception as e:
            print(f"⚠️  AI validation failed (falling back to safe): {e}")
            # Fallback: assume safe if AI fails (rule-based checks already caught dangerous patterns)
            return {'safe': True, 'issues': [], 'suggestion': None}

    @classmethod
    def validate(cls, code: str) -> Dict:
        """Complete validation pipeline"""
        result = {
            'valid': True,
            'syntax_valid': True,
            'safe': True,
            'issues': [],
            'suggestion': None
        }

        # 1. Syntax check
        syntax_valid, syntax_error = cls.validate_syntax(code)
        result['syntax_valid'] = syntax_valid
        if not syntax_valid:
            result['valid'] = False
            result['issues'].append(f"Syntax: {syntax_error}")
            return result

        # 2. Dangerous patterns
        patterns_safe, dangerous_patterns = cls.check_dangerous_patterns(code)
        if not patterns_safe:
            result['safe'] = False
            result['valid'] = False
            result['issues'].append(f"Gevaarlijke operaties gevonden: {', '.join(dangerous_patterns)}")
            result['suggestion'] = "Verwijder gevaarlijke operaties zoals file writes, subprocess calls"
            return result

        # 3. AI safety check (optional, can fail gracefully)
        ai_result = cls.ai_safety_check(code)
        if not ai_result.get('safe', True):
            result['safe'] = False
            result['valid'] = False
            result['issues'].extend(ai_result.get('issues', []))
            result['suggestion'] = ai_result.get('suggestion')

        return result


class ResponseValidator:
    """Validates AI responses for correct formatting"""

    @staticmethod
    def count_code_blocks(text: str) -> int:
        """Count CODE: prefixes in response"""
        return len(re.findall(r'\bCODE:', text))

    @staticmethod
    def validate_format(text: str) -> Tuple[bool, Optional[str]]:
        """Check if response has valid CODE: or TEXT: format"""
        if not text:
            return False, "Empty response"

        # Must start with CODE: or TEXT:
        if not (text.startswith('CODE:') or text.startswith('TEXT:')):
            return False, "Response moet beginnen met CODE: of TEXT:"

        # Check for multiple CODE: blocks
        code_count = ResponseValidator.count_code_blocks(text)
        if code_count > 1:
            return False, f"Meerdere CODE: blocks gevonden ({code_count})"

        return True, None

    @staticmethod
    def auto_fix(text: str) -> str:
        """Try to auto-fix common formatting issues"""
        if not text:
            return text

        # Remove extra CODE: blocks (keep only first)
        parts = text.split('CODE:')
        if len(parts) > 2:
            # Keep first CODE: block only
            text = 'CODE:' + parts[1].split('\n\n')[0]

        return text.strip()

    @classmethod
    def validate(cls, text: str) -> Dict:
        """Validate and optionally fix response"""
        result = {
            'valid': True,
            'issues': [],
            'fixed_text': text
        }

        valid, error = cls.validate_format(text)
        if not valid:
            # Try to auto-fix
            fixed_text = cls.auto_fix(text)
            fixed_valid, _ = cls.validate_format(fixed_text)

            if fixed_valid:
                result['fixed_text'] = fixed_text
                result['issues'].append(f"Auto-fixed: {error}")
            else:
                result['valid'] = False
                result['issues'].append(error)

        return result


class ErrorAnalyzer:
    """Analyzes errors and provides helpful explanations using AI"""

    @staticmethod
    def analyze(code: str, error_trace: str, attempt: int = 1) -> Dict:
        """Analyze error and provide tech + begrijpelijke explanations + suggestions"""
        try:
            response = openai_client.chat.completions.create(
                model=VALIDATION_CONFIG['analyzer_model'],
                messages=[{
                    "role": "system",
                    "content": """Je bent een Python error analyzer. Analyseer errors en geef:
1. Technische uitleg: wat ging er precies fout
2. Begrijpelijke uitleg: in simpel Nederlands voor niet-programmeurs
3. Suggestie: concrete stap om het op te lossen

Respond ONLY with JSON:
{
  "technical": "technische uitleg",
  "explanation": "begrijpelijke uitleg in Nederlands",
  "suggestion": "concrete actie om op te lossen",
  "recoverable": true/false
}

Wees behulpzaam en constructief."""
                }, {
                    "role": "user",
                    "content": f"""Code die faalde:
```python
{code}
```

Error (attempt {attempt}):
```
{error_trace}
```

Analyseer deze error."""
                }],
                max_completion_tokens=VALIDATION_CONFIG['analyzer_max_tokens']
            )

            result_text = response.choices[0].message.content.strip()

            # Try to parse JSON
            try:
                result = json.loads(result_text)
                result['attempt'] = attempt
                return result
            except json.JSONDecodeError:
                # Fallback: return text as-is
                return {
                    'technical': error_trace,
                    'explanation': result_text,
                    'suggestion': 'Zie de error message hierboven',
                    'recoverable': False,
                    'attempt': attempt
                }

        except Exception as e:
            print(f"⚠️  Error analysis failed: {e}")
            # Fallback: return basic error info
            return {
                'technical': error_trace,
                'explanation': 'Er ging iets fout bij het uitvoeren van de code',
                'suggestion': 'Controleer de error message voor details',
                'recoverable': False,
                'attempt': attempt
            }


class RetryHandler:
    """Manages retry logic with exponential backoff"""

    @staticmethod
    def should_retry(attempt: int, error: Exception) -> bool:
        """Determine if operation should be retried"""
        if attempt >= VALIDATION_CONFIG['max_retries']:
            return False

        # Don't retry syntax errors (already caught by validator)
        if isinstance(error, SyntaxError):
            return False

        # Don't retry dangerous operation errors
        error_msg = str(error).lower()
        if any(danger in error_msg for danger in ['permission', 'access denied', 'security']):
            return False

        return True

    @staticmethod
    def get_backoff_delay(attempt: int) -> float:
        """Calculate exponential backoff delay"""
        return min(0.1 * (2 ** attempt), 2.0)  # Max 2 seconds


class CSVInspector:
    """Inspects CSV files and detects common issues"""

    @staticmethod
    def detect_delimiter(filepath: str, max_lines: int = 10) -> str:
        """Detect the most likely delimiter"""
        delimiters = [',', ';', '\t', '|']
        delimiter_counts = {d: 0 for d in delimiters}

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    for delim in delimiters:
                        delimiter_counts[delim] += line.count(delim)

            # Return delimiter with highest count
            return max(delimiter_counts.items(), key=lambda x: x[1])[0]
        except Exception as e:
            print(f"Error detecting delimiter: {e}")
            return ','

    @staticmethod
    def find_header_row(filepath: str, delimiter: str = ',', max_lines: int = 20) -> int:
        """Find which row is most likely the header (has most non-empty values)"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [f.readline() for _ in range(max_lines)]

            max_fields = 0
            header_row = 0

            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                fields = line.split(delimiter)
                non_empty = sum(1 for field in fields if field.strip())

                if non_empty > max_fields:
                    max_fields = non_empty
                    header_row = i

            return header_row
        except Exception as e:
            print(f"Error finding header row: {e}")
            return 0

    @staticmethod
    def inspect_file(filepath: str) -> Dict:
        """Comprehensive CSV inspection"""
        report = {
            'filepath': filepath,
            'delimiter': ',',
            'header_row': 0,
            'total_lines': 0,
            'empty_lines': [],
            'encoding': 'utf-8',
            'issues': [],
            'suggestions': []
        }

        try:
            # Detect delimiter
            report['delimiter'] = CSVInspector.detect_delimiter(filepath)

            # Find header row
            report['header_row'] = CSVInspector.find_header_row(filepath, report['delimiter'])

            # Count lines and find empty ones
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        report['empty_lines'].append(i)
                    report['total_lines'] = i + 1

            # Analyze issues
            if report['header_row'] > 0:
                report['issues'].append(f"Header not on first line (found on line {report['header_row'] + 1})")
                report['suggestions'].append(f"Skip first {report['header_row']} rows using skiprows={report['header_row']}")

            if report['empty_lines']:
                report['issues'].append(f"Found {len(report['empty_lines'])} empty lines")
                report['suggestions'].append("Use skip_blank_lines=True in read_csv()")

            if report['delimiter'] != ',':
                report['issues'].append(f"Non-standard delimiter detected: '{report['delimiter']}'")
                report['suggestions'].append(f"Use sep='{report['delimiter']}' in read_csv()")

            return report

        except Exception as e:
            report['issues'].append(f"Inspection failed: {str(e)}")
            return report

    @staticmethod
    def generate_fix_code(report: Dict) -> str:
        """Generate pandas code to correctly load the CSV"""
        filepath = report['filepath']
        params = []

        if report['delimiter'] != ',':
            params.append(f"sep='{report['delimiter']}'")

        if report['header_row'] > 0:
            params.append(f"skiprows={report['header_row']}")

        if report['empty_lines']:
            params.append("skip_blank_lines=True")

        # Always use engine='python' for robustness
        params.append("engine='python'")

        params_str = ', '.join(params) if params else ''

        code = f"""import pandas as pd

# Auto-detected CSV parameters:
# - Delimiter: '{report['delimiter']}'
# - Header row: {report['header_row']}
# - Empty lines: {len(report['empty_lines'])}

df = pd.read_csv('{filepath}'{', ' + params_str if params_str else ''})

print("Dataset loaded successfully!")
print("="*50)
df.info()
print("\\n" + "="*50)
print("\\nFirst 5 rows:")
print("="*50)
print(df.head().to_html())"""

        return code


def capture_matplotlib_plots():
    """Capture matplotlib plots als base64 images"""
    plots = []
    for fignum in plt.get_fignums():
        fig = plt.figure(fignum)

        # Skip empty figures (no axes or all axes empty)
        axes = fig.get_axes()
        if not axes:
            plt.close(fig)
            continue

        # Check if any axis has actual content (lines, patches, collections, etc.)
        has_content = False
        for ax in axes:
            if (ax.lines or ax.patches or ax.collections or
                ax.images or ax.texts or ax.artists):
                has_content = True
                break

        if not has_content:
            plt.close(fig)
            continue

        # Capture the figure
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plots.append(img_base64)
        plt.close(fig)
    return plots


def capture_plotly_figures(vars_before_execution=None):
    """
    Capture Plotly figures als JSON voor interactive rendering

    Args:
        vars_before_execution: Dict mapping variable names to object IDs before execution.
                              If provided, only NEW or MODIFIED figures are captured.
                              If None, all figures are captured (legacy behavior).
    """
    if not PLOTLY_AVAILABLE:
        print("⚠️  Plotly not available")
        return []

    plotly_data = []

    print(f"🔍 Scanning {len(execution_namespace)} variables in namespace...")

    # Search execution namespace for Plotly figure objects
    for var_name, var_value in execution_namespace.items():
        # Check if it's a Plotly Figure object
        if isinstance(var_value, (go.Figure, plotly.graph_objs._figure.Figure)):
            # If we're tracking variables, skip unchanged figures from previous executions
            if vars_before_execution is not None:
                if var_name in vars_before_execution:
                    # Variable existed before - check if it's the same object
                    if id(var_value) == vars_before_execution[var_name]:
                        print(f"⏭️  Skipping unchanged Plotly figure '{var_name}' from previous execution")
                        continue
                    else:
                        print(f"🔄 Figure '{var_name}' was modified during this execution")
                else:
                    print(f"✨ Figure '{var_name}' is NEW from this execution")

            try:
                # Convert figure to JSON (Plotly's native format)
                fig_json = var_value.to_json()
                plotly_data.append(fig_json)
                var_type = type(var_value).__name__
                print(f"✓ Captured Plotly figure '{var_name}' (type: {var_type}, {len(fig_json)} bytes)")
            except Exception as e:
                print(f"⚠️  Failed to serialize Plotly figure '{var_name}': {e}")

    if plotly_data:
        print(f"📊 Total Plotly figures captured: {len(plotly_data)}")
    else:
        print("❌ No NEW Plotly figures found from this execution")

    return plotly_data

@app.route('/', methods=['GET'])
def api_info():
    """API information and available endpoints"""
    return jsonify({
        'name': 'AI Jupyter Notebook Backend API',
        'version': '5.0.0',
        'description': 'JupyterLab Edition with AI-powered code generation',
        'endpoints': {
            'GET /': 'This API information',
            'GET /health': 'Health check',
            'GET /libraries': 'List installed Python packages',
            'POST /execute': 'Execute Python code with AI validation',
            'POST /chat': 'AI chat for code generation',
            'POST /reset': 'Reset execution namespace',
            'GET /variables': 'Get current variables in namespace',
            'POST /upload': 'Upload CSV/data files',
            'GET /read-file': 'Read uploaded files',
            'POST /debug-csv': 'Debug CSV parsing issues'
        },
        'urls': {
            'backend': 'http://localhost:5000',
            'jupyterlab': 'http://localhost:8888/lab'
        }
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Backend is running'}), 200


@app.route('/restart_kernel', methods=['POST'])
def restart_kernel():
    """Restart the Python kernel - clear all variables and state"""
    global execution_namespace

    try:
        # Clear all variables from namespace
        execution_namespace.clear()

        # Close all matplotlib figures to free memory
        plt.close('all')

        return jsonify({
            'status': 'ok',
            'message': 'Kernel restarted - all variables cleared'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Kernel restart failed: {str(e)}'
        }), 500


# Removed duplicate /libraries endpoint - see line 1765 for the active implementation


@app.route('/execute', methods=['POST'])
def execute_code():
    """
    Execute Python code with validation and retry logic
    """
    try:
        data = request.get_json()
        code = data.get('code', '')

        if not code:
            return jsonify({'error': 'No code provided'}), 400

        # Remove leading whitespace from all lines
        import textwrap
        code = textwrap.dedent(code)

        # Remove fig.show() calls to prevent Plotly from trying to open browser
        import re
        code = re.sub(r'^\s*fig\.show\(\s*\)\s*$', '# fig.show() removed - backend auto-captures Plotly figures', code, flags=re.MULTILINE)

        # ===== VALIDATION PIPELINE =====
        print(f"\n🔍 Validating code ({len(code)} chars)...")
        validation_result = CodeValidator.validate(code)

        if not validation_result['valid']:
            # Return validation error
            error_info = {
                'technical': f"Validation failed: {', '.join(validation_result['issues'])}",
                'explanation': 'De code bevat fouten of onveilige operaties',
                'suggestion': validation_result.get('suggestion', 'Controleer de code en probeer opnieuw'),
                'recoverable': False,
                'attempt': 0
            }
            return jsonify({
                'output': None,
                'plots': [],
                'plotly_data': [],
                'error': error_info
            }), 200

        print("✓ Validation passed")

        # ===== EXECUTION WITH RETRY =====
        last_error = None
        last_error_trace = None

        for attempt in range(1, VALIDATION_CONFIG['max_retries'] + 1):
            print(f"\n▶ Execution attempt {attempt}/{VALIDATION_CONFIG['max_retries']}")

            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()

            try:
                # Configure Plotly to NOT open browser (prevents blocking in headless environments)
                if '__plotly_configured__' not in execution_namespace:
                    import plotly.io as pio
                    pio.renderers.default = 'json'  # Don't try to open browser
                    execution_namespace['__plotly_configured__'] = True

                # 🔍 TRACK VARIABLES BEFORE EXECUTION (to detect new/modified Plotly figures)
                vars_before_execution = {k: id(v) for k, v in execution_namespace.items()}

                # Execute code with Jupyter-like behavior: auto-print last expression
                with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                    # Try to detect if last line is an expression (not assignment/statement)
                    # and auto-print its result like Jupyter does
                    try:
                        # Parse code to AST
                        tree = ast.parse(code, mode='exec')

                        if tree.body and isinstance(tree.body[-1], ast.Expr):
                            # Last statement is an expression - split it out
                            last_expr = tree.body[-1]
                            previous_stmts = tree.body[:-1]

                            # Execute all statements except the last
                            if previous_stmts:
                                previous_tree = ast.Module(body=previous_stmts, type_ignores=[])
                                exec(compile(previous_tree, '<string>', 'exec'), execution_namespace)

                            # Evaluate the last expression and print if not None
                            last_expr_code = compile(ast.Expression(body=last_expr.value), '<string>', 'eval')
                            result_value = eval(last_expr_code, execution_namespace)

                            if result_value is not None:
                                # Use IPython-like display for DataFrames, or regular print
                                import pandas as pd

                                # Skip printing Plotly figures - they're captured separately
                                if PLOTLY_AVAILABLE and isinstance(result_value, (go.Figure, plotly.graph_objs._figure.Figure)):
                                    pass  # Don't print - will be captured by capture_plotly_figures()
                                elif isinstance(result_value, pd.DataFrame):
                                    print(result_value.to_string())
                                elif isinstance(result_value, pd.Series):
                                    print(result_value.to_string())
                                else:
                                    print(repr(result_value))
                        else:
                            # No expression at end, or parsing failed - just exec normally
                            exec(code, execution_namespace)
                    except (SyntaxError, ValueError):
                        # Fallback to normal exec if AST parsing fails
                        exec(code, execution_namespace)

                # Success! Get output
                output = stdout_capture.getvalue()
                error_output = stderr_capture.getvalue()
                plots = capture_matplotlib_plots()
                # Only capture NEW or MODIFIED Plotly figures from this execution
                plotly_data = capture_plotly_figures(vars_before_execution)

                result = output
                if error_output:
                    result += '\nWarnings/Errors:\n' + error_output

                total_visuals = len(plots) + len(plotly_data)
                if not result and total_visuals:
                    result = f'{len(plots)} matplotlib plot{"s" if len(plots) != 1 else ""}, {len(plotly_data)} interactive plot{"s" if len(plotly_data) != 1 else ""} generated'
                elif not result:
                    result = 'Code executed successfully (no output)'

                print(f"✓ Execution succeeded on attempt {attempt}")
                if plotly_data:
                    print(f"  Captured {len(plotly_data)} Plotly figure(s)")

                return jsonify({
                    'output': result,
                    'plots': plots,
                    'plotly_data': plotly_data,
                    'error': None,
                    'attempt': attempt
                }), 200

            except Exception as e:
                last_error = e
                last_error_trace = traceback.format_exc()

                print(f"✗ Attempt {attempt} failed: {type(e).__name__}: {str(e)[:100]}")

                # Check if we should retry
                if attempt < VALIDATION_CONFIG['max_retries'] and RetryHandler.should_retry(attempt, e):
                    # Wait with exponential backoff
                    delay = RetryHandler.get_backoff_delay(attempt)
                    print(f"  Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    continue
                else:
                    # No more retries, analyze error
                    break

        # ===== ERROR ANALYSIS =====
        print(f"\n⚠️  All attempts failed. Analyzing error...")

        error_analysis = ErrorAnalyzer.analyze(
            code=code,
            error_trace=last_error_trace,
            attempt=attempt
        )

        print(f"📊 Error analysis complete")
        print(f"   Technical: {error_analysis.get('technical', '')[:100]}...")
        print(f"   Explanation: {error_analysis.get('explanation', '')[:100]}...")

        return jsonify({
            'output': None,
            'plots': [],
            'plotly_data': [],
            'error': error_analysis
        }), 200

    except Exception as e:
        # Server error (shouldn't happen)
        error_trace = traceback.format_exc()
        print(f"\n❌ Server error: {error_trace}")

        return jsonify({
            'output': None,
            'plots': [],
            'plotly_data': [],
            'error': {
                'technical': error_trace,
                'explanation': 'Er is een server error opgetreden',
                'suggestion': 'Probeer de code opnieuw uit te voeren',
                'recoverable': True,
                'attempt': 0
            }
        }), 500


@app.route('/reset', methods=['POST'])
def reset_namespace():
    """Reset de execution namespace"""
    global execution_namespace
    execution_namespace = {}
    return jsonify({'message': 'Namespace reset successfully'}), 200


@app.route('/variables', methods=['GET'])
def get_variables():
    """Get alle variabelen in de current namespace"""
    # Filter out built-in variables
    user_vars = {k: str(v) for k, v in execution_namespace.items()
                 if not k.startswith('__')}
    return jsonify({'variables': user_vars}), 200


@app.route('/export/py', methods=['POST'])
def export_py():
    """Export all cells as Python file"""
    try:
        data = request.get_json()
        cells = data.get('cells', [])

        # Combine all cell code with markers
        py_content = "# AI Notebook Export\n"
        py_content += "# Generated by AI Notebook v5\n\n"

        for i, cell in enumerate(cells, 1):
            py_content += f"# ========== Cell {i} ==========\n"
            py_content += cell.get('code', '') + "\n\n"

        # Return as downloadable file
        response = make_response(py_content)
        response.headers['Content-Type'] = 'text/plain'
        response.headers['Content-Disposition'] = 'attachment; filename=notebook.py'
        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/export/ipynb', methods=['POST'])
def export_ipynb():
    """Export all cells as Jupyter notebook with outputs"""
    try:
        data = request.get_json()
        cells = data.get('cells', [])

        # Build Jupyter notebook structure
        nb_cells = []
        for cell_data in cells:
            nb_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [cell_data.get('code', '')]
            }

            # Add output if exists
            output = cell_data.get('output', '')
            if output:
                nb_cell["outputs"] = [{
                    "output_type": "stream",
                    "name": "stdout",
                    "text": [output]
                }]
            else:
                nb_cell["outputs"] = []

            nb_cells.append(nb_cell)

        # Create full notebook structure
        notebook = {
            "cells": nb_cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.12.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }

        # Return as downloadable file
        response = make_response(json.dumps(notebook, indent=2))
        response.headers['Content-Type'] = 'application/json'
        response.headers['Content-Disposition'] = 'attachment; filename=notebook.ipynb'
        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/import/py', methods=['POST'])
def import_py():
    """Import Python file and split into cells"""
    try:
        data = request.get_json()
        content = data.get('content', '')

        # Split by cell markers or by function/class definitions
        cells = []

        # First try to split by our export markers
        if '# ========== Cell' in content:
            parts = re.split(r'# ========== Cell \d+ ==========\n', content)
            for part in parts:
                code = part.strip()
                if code and not code.startswith('# AI Notebook Export'):
                    cells.append({'code': code})
        else:
            # No markers, split by double newlines or keep as single cell
            parts = content.split('\n\n')
            if len(parts) > 1:
                for part in parts:
                    code = part.strip()
                    if code:
                        cells.append({'code': code})
            else:
                # Keep as single cell
                cells.append({'code': content.strip()})

        return jsonify({'cells': cells}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/import/ipynb', methods=['POST'])
def import_ipynb():
    """Import Jupyter notebook with outputs"""
    try:
        data = request.get_json()
        content = data.get('content', '')

        # Parse Jupyter notebook JSON
        notebook = json.loads(content)
        cells = []

        for nb_cell in notebook.get('cells', []):
            if nb_cell.get('cell_type') == 'code':
                # Get source code
                source = nb_cell.get('source', [])
                if isinstance(source, list):
                    code = ''.join(source)
                else:
                    code = source

                # Get output
                output_text = ''
                outputs = nb_cell.get('outputs', [])
                for output in outputs:
                    if output.get('output_type') == 'stream':
                        text = output.get('text', [])
                        if isinstance(text, list):
                            output_text += ''.join(text)
                        else:
                            output_text += text
                    elif output.get('output_type') == 'execute_result':
                        data_output = output.get('data', {})
                        if 'text/plain' in data_output:
                            text = data_output['text/plain']
                            if isinstance(text, list):
                                output_text += ''.join(text)
                            else:
                                output_text += text

                cells.append({
                    'code': code.strip(),
                    'output': output_text.strip()
                })

        return jsonify({'cells': cells}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat with OpenAI GPT-4o mini
    """
    try:
        data = request.get_json()
        message = data.get('message', '')
        history = data.get('history', [])
        recent_cells = data.get('recent_cells', [])
        uploaded_file = data.get('uploaded_file', None)

        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Check if API key is configured
        if not os.getenv('OPENAI_API_KEY'):
            return jsonify({
                'error': 'OpenAI API key not configured. Please add OPENAI_API_KEY to .env file'
            }), 500

        # ===== CSV DEBUG DETECTION =====
        message_lower = message.lower()
        is_csv_debug = any(keyword in message_lower for keyword in ['debug csv', 'fix csv', 'debug my csv', 'fix my csv', 'csv probleem', 'csv error'])

        if is_csv_debug:
            print(f"\n🔍 CSV debug request detected: '{message}'")

            # Find most recent CSV file in uploads folder
            csv_files = []
            for filename in os.listdir(UPLOAD_FOLDER):
                if filename.lower().endswith('.csv'):
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    mtime = os.path.getmtime(filepath)
                    csv_files.append((filepath, mtime))

            if not csv_files:
                return jsonify({
                    'message': 'No CSV files found. Please upload a CSV file first.',
                    'type': 'text',
                    'model': 'system',
                    'error': None
                }), 200

            # Get most recent CSV file
            latest_csv = sorted(csv_files, key=lambda x: x[1], reverse=True)[0][0]
            print(f"   Debugging file: {latest_csv}")

            # Inspect the CSV file
            report = CSVInspector.inspect_file(latest_csv)

            if not report['issues']:
                # No issues found - file is clean
                return jsonify({
                    'message': f"No issues found in {os.path.basename(latest_csv)}! The file looks clean and can be loaded normally with pd.read_csv().",
                    'type': 'text',
                    'model': 'csv-inspector',
                    'error': None
                }), 200

            # Generate fix code
            fix_code = CSVInspector.generate_fix_code(report)

            print(f"   Issues found: {len(report['issues'])}")
            for issue in report['issues']:
                print(f"   - {issue}")

            # Return the fix code directly
            return jsonify({
                'message': fix_code,
                'type': 'code',
                'model': 'csv-inspector',
                'error': None
            }), 200

        # Build messages array with history
        messages = [
            {"role": "system", "content": """You are a Python data analysis assistant in a Jupyter notebook with pandas, matplotlib, and plotly support.

CORE PRINCIPLE: **MINIMALISM** - Generate the SHORTEST possible code that does EXACTLY what the user asks. Nothing more.

CRITICAL DECISION RULE:
- If user asks to CREATE, MAKE, BUILD, DO, VISUALIZE, PLOT, SHOW, CALCULATE something → Respond with CODE:
- If user asks to EXPLAIN existing code/output → Respond with TEXT: (analyze the provided code)
- If user is just TALKING (greeting, general questions, chatting, saying thanks) → Respond with TEXT:

IMPORTANT: When user says "maak" (Dutch for "make"), "create", "build" → ALWAYS generate CODE, not explanation!

CODE GENERATION RULES (only when user wants to DO something):
1. Start with CODE: followed by executable Python code
2. ONLY ONE CODE: block per response - never include multiple CODE: statements
3. ⚠️ GENERATE MINIMAL CODE:
   - Write the SHORTEST code that fulfills the request
   - DO NOT add extra features, error handling, or robustness unless EXPLICITLY requested
   - DO NOT add print statements, comments, or explanations unless asked
   - Use simple, direct solutions - avoid complexity
   - If user asks "load CSV" → Just use pd.read_csv(filepath) - don't inspect, analyze, or debug
   - If user asks "plot X" → Just plot X - don't add titles, grids, styling unless asked
   - If user asks "calculate Y" → Just calculate Y - don't print extra info unless asked

   EXAMPLES OF MINIMAL CODE:
   ❌ WRONG (too much):
   ```python
   import pandas as pd
   print("Loading data...")
   df = pd.read_csv(filepath, encoding='utf-8', errors='ignore')
   print(f"Loaded {len(df)} rows")
   print("="*50)
   df.info()
   print("="*50)
   print(df.head())
   ```

   ✅ CORRECT (minimal):
   ```python
   df = pd.read_csv(filepath)
   df.head()
   ```

4. ⚠️ CRITICAL: ALWAYS CHECK AVAILABLE CONTEXT FIRST!
   Before writing ANY code, look for "AVAILABLE CONTEXT:" section in the prompt.

   If you see "Available variables: df_iris (DataFrame: 150 rows x 5 cols)"
   → DO NOT load iris again! Use the existing df_iris variable!

   If you see "Already imported modules: pandas, sklearn"
   → DO NOT import pandas or sklearn again! They're already available!

   EXAMPLES:
   ❌ WRONG: Load iris again when df_iris exists
   from sklearn.datasets import load_iris
   iris = load_iris()

   ✅ CORRECT: Use existing df_iris
   fig = go.Figure(data=[go.Scatter(x=df_iris['sepal length (cm)'], y=df_iris['sepal width (cm)'])])

   ❌ WRONG: Import pandas when already imported
   import pandas as pd
   df = pd.read_csv(...)

   ✅ CORRECT: Just use pd directly (it's already imported)
   df = pd.read_csv(...)

4. For datasets (iris, titanic, etc.), ONLY load them if NO DataFrame variable exists yet
5. For INTERACTIVE visualizations (3D plots, zoom/pan), use plotly.graph_objects - END CODE AFTER fig = ... (no fig.show()!)
6. For STATIC visualizations, use matplotlib with plt.show()
   ⚠️ CRITICAL PLOTLY RULE: After creating Plotly figure, do NOT add fig.show() - backend captures 'fig' automatically!
7. Import only NEW libraries that aren't already imported (check "Already imported modules:" in context)
8. Do NOT add comments or explanations
9. Return ONLY the code, nothing else after CODE:
10. For yfinance: Use Ticker().history() instead of download() to avoid MultiIndex:
    CORRECT: ticker = yf.Ticker('BTC-USD'); data = ticker.history(period='90d', auto_adjust=True)
    WRONG: data = yf.download('BTC-USD', period='90d', auto_adjust=True)  # Creates MultiIndex!
11. After yfinance download, ALWAYS check if data is empty before using
12. For yfinance datetime access: use data.index directly, NOT reset_index() and data['Date']
    CORRECT: fig.add_trace(go.Scatter(x=data.index, y=data['Close']))
    WRONG: data = data.reset_index(); fig.add_trace(go.Scatter(x=data['Date'], y=data['Close']))
13. For matplotlib bar charts: MUST use .values to convert pandas Series to numpy array
    CORRECT: ax.bar(range(len(df)), df['Volume'].values, color='#888888', alpha=0.7)
    WRONG: ax.bar(range(len(df)), df['Volume'], ...) - this causes TypeError with pandas Series
14. ⚠️ SEABORN FIGURE-LEVEL FUNCTIONS: pairplot(), catplot(), lmplot(), relplot() create their OWN figure!
    CORRECT: sns.pairplot(df, hue='species')  # No plt.figure() before!
    WRONG: plt.figure(figsize=(8,8)); sns.pairplot(df, hue='species')  # Creates empty figure!
    REASON: matplotlib can't handle pandas Series with DatetimeIndex directly, need .values
15. NEVER use if __name__ == '__main__': - code is executed via exec() so this condition is always False
16. Always call your main functions directly at the end, do NOT wrap in if __name__ check

CONVERSATIONAL RULES (when user is just talking):
1. Start with TEXT: followed by your response
2. Be helpful and friendly
3. Answer questions about concepts, give advice
4. ALWAYS use markdown formatting in your TEXT responses:
   - Use code blocks with language specification: ```python for Python code examples
   - Use **bold** for emphasis
   - Use headers (##, ###) to organize content
   - Use lists (-, *) for bullet points
   - Use numbered lists (1., 2., 3.) for sequential steps
   - Use `inline code` for variable names, function names, and short code snippets
   - Use > for important notes or quotes
   - Use ASCII art and text visualizations to illustrate concepts:
     * Box drawings: ┌─────┐  │     │  └─────┘
     * Arrows: ─→ ←─ ↓ ↑
     * Simple diagrams with |, -, +, /, \
     * Tables and structured layouts
     Example: Create flowcharts, diagrams, timelines using ASCII characters when explaining processes

Examples:

User: "hello"
Assistant: TEXT:Hello! I'm your data analysis assistant. Upload a CSV/Excel file to get started, or ask me anything about Python and data analysis.

User: "test"
Assistant: TEXT:I'm here and ready to help! Upload a dataset and I can help you visualize and analyze it.

User: "maak hello world"
Assistant: CODE:print("Hello World")

User: "maak een grafiek"
Assistant: CODE:import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title('Sinus Golf')
plt.show()

User: "thanks"
Assistant: TEXT:You're welcome! Let me know if you need anything else.

User: "what is pandas?"
Assistant: TEXT:Pandas is a powerful Python library for data manipulation and analysis. It provides DataFrame structures for working with tabular data.

User: "visualize iris dataset"
Assistant: CODE:from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['species'] = iris.target
plt.figure(figsize=(10, 6))
plt.scatter(df_iris[iris.feature_names[0]], df_iris[iris.feature_names[1]], c=df_iris['species'], cmap='viridis', alpha=0.6)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('Iris Dataset')
plt.colorbar(label='Species')
plt.show()

User: "load titanic dataset and show info"
Assistant: CODE:df = sns.load_dataset('titanic')
df.info()
df.head()

User: "visualize titanic dataset"
Assistant: CODE:import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = sns.load_dataset('titanic')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.countplot(x='pclass', hue='survived', data=df, ax=axes[0,0])
axes[0,0].set_title('Survival by Class')
sns.countplot(x='sex', hue='survived', data=df, ax=axes[0,1])
axes[0,1].set_title('Survival by Sex')
sns.histplot(data=df, x='age', hue='survived', bins=30, ax=axes[1,0])
axes[1,0].set_title('Age Distribution')
sns.boxplot(x='survived', y='fare', data=df, ax=axes[1,1])
axes[1,1].set_title('Fare by Survival')
plt.tight_layout()
plt.show()

User: "visualize gender distribution"
Assistant: CODE:import matplotlib.pyplot as plt
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(gender_counts.index, gender_counts.values, color=['#4ec9b0', '#569cd6'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

User: "show age distribution"
Assistant: CODE:import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=20, color='#4ec9b0', edgecolor='black', alpha=0.7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

User: "calculate average age"
Assistant: CODE:df['Age'].mean()

User: "visualiseer alle verdelingen"
Assistant: CODE:import matplotlib.pyplot as plt
import numpy as np
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=20, color='#4ec9b0', edgecolor='black', alpha=0.7)
    axes[i].set_title(f'{col} Distribution')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

User: "correlation heatmap"
Assistant: CODE:import matplotlib.pyplot as plt
import seaborn as sns
numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

User: "make starter code"
Assistant: CODE:data = {'x': [1, 2, 3, 4, 5], 'y': [2, 4, 6, 8, 10]}
df = pd.DataFrame(data)
df

User: "visualize process flow"
Assistant: CODE:print('''flowchart TD
  A["Start"] --> B["Process Step 1"]
  B --> C{"Decision?"}
  C -->|"Yes"| D["Step 2A"]
  C -->|"No"| E["Step 2B"]
  D --> F["End"]
  E --> F''')

User: "create onboarding diagram"
Assistant: CODE:diagram = '''flowchart LR
  A["New Employee"] --> B["HR Onboarding"]
  B --> C["Equipment Setup"]
  C --> D["Team Introduction"]
  D --> E["Training"]
  E --> F["First Project"]'''
print(diagram)

User: "visualize bitcoin price"
Assistant: CODE:import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
btc = yf.download('BTC-USD', period='90d', auto_adjust=True)
if btc.empty:
    print('No data received')
else:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(btc.index, btc['Close'], label='BTC-USD', color='#4ec9b0', linewidth=2)
    ax1.set_title('Bitcoin Price (90 days)')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.bar(range(len(btc)), btc['Volume'], color='#888888', alpha=0.7)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    ax2.set_xticks(range(0, len(btc), max(1, len(btc)//10)))
    ax2.set_xticklabels([btc.index[i].strftime('%Y-%m-%d') for i in range(0, len(btc), max(1, len(btc)//10))], rotation=45)
    plt.tight_layout()
    plt.show()
    print(f'Latest: ${btc["Close"].iloc[-1]:,.2f}')

User: "load ethereum data"
Assistant: CODE:import yfinance as yf
eth = yf.download('ETH-USD', period='30d', auto_adjust=True)
if not eth.empty:
    print(f'ETH data loaded: {len(eth)} days')
    print(f'Latest price: ${eth["Close"].iloc[-1]:,.2f}')
    print(eth.tail())
else:
    print('Failed to load ETH data')

User: "visualize bitcoin with plotly"
Assistant: CODE:import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
btc = yf.Ticker('BTC-USD')
data = btc.history(period='90d', auto_adjust=True)
if not data.empty:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='BTC'), row=1, col=1)
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'), row=2, col=1)
    fig.update_layout(height=600, title_text='Bitcoin - Last 90 Days')
    print(f'Latest: ${data["Close"].iloc[-1]:,.2f}')
else:
    print('No data')

User: "3d scatter plot with iris"
Assistant: CODE:import plotly.graph_objects as go
from sklearn.datasets import load_iris
iris = load_iris()
fig = go.Figure(data=[go.Scatter3d(x=iris.data[:,0], y=iris.data[:,1], z=iris.data[:,2], mode='markers', marker=dict(size=5, color=iris.target, colorscale='Viridis'))])
fig.update_layout(title='Iris 3D Scatter', scene=dict(xaxis_title='Sepal Length', yaxis_title='Sepal Width', zaxis_title='Petal Length'))"""}
        ]

        # Add history if provided (last few messages for context)
        if history:
            messages.extend(history[-3:])  # Last 3 messages max

        # Extract imports from recent cells code
        # Separate tracking for full module imports vs from imports
        full_module_imports = set()  # Modules imported with 'import X' or 'import X as Y'
        from_imports = {}  # Tracks 'from X import Y' - stores {module: [items]}

        if recent_cells:
            for cell in recent_cells:
                if cell['type'] == 'code' and cell['code']:
                    # Extract imports using regex
                    code_lines = cell['code'].split('\n')
                    for line in code_lines:
                        line = line.strip()
                        # Match: import pandas, import pandas as pd, from sklearn import ...
                        if line.startswith('import ') and ' from ' not in line:
                            # Full module import: import pandas, import pandas as pd
                            import_part = line.replace('import ', '')
                            if ' as ' in import_part:
                                # import pandas as pd → add 'pd' as available
                                parts = import_part.split(' as ')
                                if len(parts) >= 2:
                                    alias = parts[1].split()[0].replace(',', '')
                                    full_module_imports.add(alias)
                            else:
                                # import pandas, numpy → add pandas, numpy as available
                                modules = import_part.split(',')
                                for mod in modules:
                                    mod_name = mod.strip().split('.')[0]  # sklearn.tree → sklearn
                                    full_module_imports.add(mod_name)

                        elif line.startswith('from '):
                            # from X import Y - X is NOT available directly
                            parts = line.split(' import ')
                            if len(parts) >= 2:
                                module_part = parts[0].replace('from ', '').strip()
                                imports_part = parts[1].strip()

                                # Get base module name
                                base_module = module_part.split('.')[0]

                                # Track what was imported FROM this module
                                if base_module not in from_imports:
                                    from_imports[base_module] = []

                                # Add imported items
                                import_items = [item.strip() for item in imports_part.split(',')]
                                from_imports[base_module].extend(import_items)

        # Add recent cells context if provided
        context_message = message
        if recent_cells:
            context_parts = []
            for cell in recent_cells:
                if cell['type'] == 'code' and cell['code']:
                    context_parts.append(f"Previous code:\n{cell['code']}")
                    if cell.get('output'):  # Use .get() to avoid KeyError
                        context_parts.append(f"Output/Error:\n{cell['output']}")

            if context_parts:
                context = "\n\n".join(context_parts)
                context_message = f"{context}\n\nUser question: {message}"

        # Add info about available variables and modules in execution namespace
        namespace_info = ""
        variables = []
        imported_modules = []

        # Scan execution namespace for variables and modules
        if execution_namespace:
            for name, value in execution_namespace.items():
                if name.startswith('_'):
                    continue

                # Check if it's a module
                if hasattr(value, '__name__') and hasattr(value, '__file__'):
                    imported_modules.append(name)
                # Check if it's a valuable variable (DataFrame, list, dict, etc)
                elif not callable(value) or name in ['df', 'data', 'result']:
                    var_type = type(value).__name__
                    # Add size info for DataFrames
                    if var_type == 'DataFrame':
                        try:
                            shape = value.shape
                            variables.append(f"{name} (DataFrame: {shape[0]} rows x {shape[1]} cols)")
                        except:
                            variables.append(f"{name} ({var_type})")
                    elif var_type in ['list', 'dict', 'tuple', 'set']:
                        try:
                            size = len(value)
                            variables.append(f"{name} ({var_type} with {size} items)")
                        except:
                            variables.append(f"{name} ({var_type})")
                    else:
                        variables.append(f"{name} ({var_type})")

        # Combine imports from namespace and from code analysis
        # Modules from namespace take priority (they're actually in memory)
        all_full_imports = set(imported_modules) | full_module_imports

        # Build context message - ALWAYS add it if there are imports or variables
        # This ensures AI knows about code-level imports even if namespace is empty (e.g., after restart)
        if variables or all_full_imports or from_imports:
            namespace_info = f"\n\nAVAILABLE CONTEXT:\n"

            if all_full_imports:
                namespace_info += f"Fully imported modules (can use module.function):\n"
                namespace_info += f"  {', '.join(sorted(all_full_imports))}\n"
                namespace_info += f"⚠️ CRITICAL: DO NOT re-import these! They are already available.\n\n"

            if from_imports:
                namespace_info += f"Partial imports (from module import ...):\n"
                for module, items in sorted(from_imports.items()):
                    namespace_info += f"  from {module}: {', '.join(items)}\n"
                namespace_info += f"\n⚠️ WARNING: These base modules ({', '.join(sorted(from_imports.keys()))}) are NOT directly accessible!\n"
                namespace_info += f"You must either:\n"
                namespace_info += f"  1. Use the imported items directly (e.g., 'load_iris()' not 'sklearn.datasets.load_iris()')\n"
                namespace_info += f"  2. Or import the full module first (e.g., 'import sklearn.model_selection')\n\n"

            if variables:
                namespace_info += f"Available variables from previous cells:\n"
                for var in variables[:10]:  # Limit to 10 most relevant
                    namespace_info += f"  - {var}\n"
                namespace_info += f"\nIMPORTANT: Use these existing variables! Don't reload data that's already available.\n"

            context_message = context_message + namespace_info

        # Add uploaded file info (NOT the content - for privacy and token savings!)
        if uploaded_file:
            file_info = f"\n\nUPLOADED FILE INFO:\n"
            file_info += f"- Filename: {uploaded_file['filename']}\n"
            file_info += f"- File path on server: {uploaded_file['filepath']}\n"
            file_info += f"- File type: {uploaded_file['extension']}\n"
            file_info += f"\nIMPORTANT INSTRUCTIONS FOR FUZZY FILE HANDLING:\n"
            file_info += f"1. Use this exact filepath in your code: {uploaded_file['filepath']}\n"
            file_info += f"2. The file is already on the server - DO NOT ask to see the content\n"
            file_info += f"3. CSV/Excel files are often 'fuzzy' - they may have:\n"
            file_info += f"   - Junk/test data in first few rows\n"
            file_info += f"   - Wrong delimiter (semicolon instead of comma, or vice versa)\n"
            file_info += f"   - Prefix characters like ';;' before each line\n"
            file_info += f"   - Headers on wrong row\n"
            file_info += f"   - BOM characters or encoding issues\n"
            file_info += f"\n4. CSV/EXCEL LOADING - Keep it SIMPLE:\n"
            file_info += f"   - First try: df = pd.read_csv(filepath) or df = pd.read_excel(filepath)\n"
            file_info += f"   - If user mentions problems, THEN add parameters (sep, skiprows, etc)\n"
            file_info += f"   - DO NOT proactively inspect/analyze/debug files unless user asks\n"
            file_info += f"   - Show df.head() to verify loading if user asks to load data\n"
            file_info += f"\n5. If loading fails, user can ask \"debug csv\" to get automatic analysis"
            context_message = context_message + file_info

        # Add current message
        messages.append({"role": "user", "content": context_message})

        # Try GPT-5 nano first (cheapest)
        model_used = 'gpt-5-nano'
        response = openai_client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            max_completion_tokens=4000
        )

        assistant_message = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        # Debug logging
        print("\n" + "="*50)
        print("DEBUG: OpenAI Response")
        print("="*50)
        print(f"Model: {model_used}")
        print(f"Finish reason: {finish_reason}")
        print(f"Assistant message: {repr(assistant_message)}")
        print(f"Message length: {len(assistant_message) if assistant_message else 0}")
        print("="*50 + "\n")

        # Fallback to GPT-5 mini if nano gave empty response or hit length limit
        if not assistant_message or (finish_reason == 'length' and not assistant_message):
            print("⚠️  GPT-5 nano returned empty response, falling back to GPT-5 mini...")
            model_used = 'gpt-5-mini'
            response = openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages,
                max_completion_tokens=4000
            )
            assistant_message = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Debug logging for fallback
            print("\n" + "="*50)
            print("DEBUG: Fallback Response (GPT-5 mini)")
            print("="*50)
            print(f"Finish reason: {finish_reason}")
            print(f"Assistant message: {repr(assistant_message)}")
            print(f"Message length: {len(assistant_message) if assistant_message else 0}")
            print("="*50 + "\n")

        # ===== RESPONSE VALIDATION =====
        content = assistant_message if assistant_message else ''

        if content:  # Only validate if we have content
            validation_result = ResponseValidator.validate(content)

            if not validation_result['valid']:
                print(f"⚠️  Response validation failed: {validation_result['issues']}")
                # Could not fix, return error
                return jsonify({
                    'message': None,
                    'error': f"AI response formatting error: {', '.join(validation_result['issues'])}",
                    'model': model_used
                }), 200

            # Use fixed text if validation auto-fixed issues
            if validation_result.get('issues'):
                print(f"✓ Response auto-fixed: {validation_result['issues']}")

            content = validation_result['fixed_text']

        # Parse response type
        response_type = 'text'

        if content.startswith('CODE:'):
            response_type = 'code'
            content = content[5:].strip()
        elif content.startswith('TEXT:'):
            content = content[5:].strip()

        return jsonify({
            'message': content,
            'type': response_type,
            'model': model_used,
            'error': None
        }), 200

    except Exception as e:
        error_trace = traceback.format_exc()
        return jsonify({
            'message': None,
            'error': f'Chat error: {str(e)}\n{error_trace}'
        }), 500


@app.route('/debug-csv', methods=['POST'])
def debug_csv():
    """
    Debug a CSV file and generate fix code
    """
    try:
        data = request.get_json()
        filepath = data.get('filepath', '')

        if not filepath:
            return jsonify({'error': 'No filepath provided'}), 400

        if not os.path.exists(filepath):
            return jsonify({'error': f'File not found: {filepath}'}), 404

        # Inspect the CSV file
        print(f"\n🔍 Inspecting CSV file: {filepath}")
        report = CSVInspector.inspect_file(filepath)

        # Generate fix code
        fix_code = CSVInspector.generate_fix_code(report)

        print(f"✓ CSV inspection complete")
        print(f"   Issues found: {len(report['issues'])}")
        for issue in report['issues']:
            print(f"   - {issue}")

        return jsonify({
            'report': report,
            'fix_code': fix_code,
            'error': None
        }), 200

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"❌ CSV debug error: {error_trace}")
        return jsonify({
            'error': f'CSV debug error: {str(e)}',
            'report': None,
            'fix_code': None
        }), 500


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Upload CSV or Excel file
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Only CSV, Excel, Python (.py), and Jupyter Notebook (.ipynb) files'}), 400

        # Save file securely
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get file extension
        extension = filename.rsplit('.', 1)[1].lower()

        return jsonify({
            'filename': filename,
            'filepath': filepath,
            'extension': extension,
            'message': 'File uploaded successfully'
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Upload error: {str(e)}'
        }), 500


@app.route('/libraries', methods=['GET'])
def get_libraries():
    """
    Get list of ALL installed Python packages with versions
    """
    try:
        import pkg_resources

        # Get all installed packages
        installed_packages = []
        for package in pkg_resources.working_set:
            installed_packages.append({
                'name': package.project_name,
                'version': package.version
            })

        # Sort by name
        installed_packages.sort(key=lambda x: x['name'].lower())

        return jsonify({
            'status': 'ok',
            'count': len(installed_packages),
            'packages': installed_packages
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Error retrieving libraries: {str(e)}',
            'packages': []
        }), 500


@app.route('/read-file', methods=['GET'])
def read_file():
    """
    Read content of uploaded file (for .py and .ipynb files)
    """
    try:
        filepath = request.args.get('path', '')

        if not filepath:
            return jsonify({'error': 'No filepath provided'}), 400

        # Security check: only allow reading from uploads folder
        if not filepath.startswith(UPLOAD_FOLDER):
            return jsonify({'error': 'Access denied: can only read from uploads folder'}), 403

        if not os.path.exists(filepath):
            return jsonify({'error': f'File not found: {filepath}'}), 404

        # Read file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        return jsonify({
            'content': content,
            'filepath': filepath
        }), 200

    except Exception as e:
        return jsonify({
            'error': f'Read error: {str(e)}'
        }), 500


@app.route('/suggestions', methods=['POST'])
def generate_suggestions():
    """
    AI-powered context-aware suggestions
    """
    try:
        data = request.get_json()
        last_message = data.get('last_message', '')
        recent_cells = data.get('recent_cells', [])
        uploaded_file = data.get('uploaded_file', None)

        # Build context for AI
        context = f"Last AI response: {last_message}\n\n"

        if uploaded_file:
            context += f"User uploaded: {uploaded_file.get('filename', 'a file')}\n"

        if recent_cells:
            context += f"Recent code cells: {len(recent_cells)} cells executed\n"
            # Include last cell code if exists
            for cell in recent_cells[-2:]:
                if cell.get('type') == 'code' and cell.get('code'):
                    context += f"- {cell['code'][:100]}...\n"

        # Ask AI for suggestions
        response = openai_client.chat.completions.create(
            model='gpt-5-nano',
            messages=[{
                "role": "system",
                "content": """Generate 3-4 smart follow-up suggestions for the user based on context.

RULES:
- Suggestions should be SHORT (2-4 words max)
- Must logically continue the conversation
- Consider what data exists, what was just discussed
- Be specific and actionable

Return ONLY a JSON array of strings:
["suggestion 1", "suggestion 2", "suggestion 3"]

Examples:
- After loading CSV: ["Show first rows", "Check for nulls", "Plot distribution"]
- After visualization: ["Add more styling", "Export as image", "Try 3D version"]
- After calculation: ["Visualize result", "Compare with average", "Round to 2 decimals"]"""
            }, {
                "role": "user",
                "content": context
            }],
            max_completion_tokens=200
        )

        suggestions_text = response.choices[0].message.content.strip()

        # Parse JSON
        try:
            suggestions = json.loads(suggestions_text)
            if not isinstance(suggestions, list):
                suggestions = []
        except:
            # Fallback: basic suggestions
            suggestions = ["Continue", "Explain more", "Show example"]

        return jsonify({
            'suggestions': suggestions[:4]  # Max 4
        }), 200

    except Exception as e:
        print(f"Suggestion generation error: {e}")
        # Fallback suggestions
        return jsonify({
            'suggestions': ["Continue", "Next step", "Show more"]
        }), 200


if __name__ == '__main__':
    print("=" * 50)
    print("AI Jupyter Notebook Backend")
    print("=" * 50)
    print("Backend draait op: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("\nDruk Ctrl+C om te stoppen")
    print("=" * 50)

    # Start Flask server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True
    )
