# AI Based File Prefetcher System

This project is an advanced system-level file prefetcher that uses an LSTM model to predict and aggressively preload files into the OS cache before an application needs them. It automatically traces file accesses, processes the logs into integer sequences, trains an LSTM model, and evaluates its own performance via caching speeds and metrics.

## Running the Project

**1. Setup Environment**
Make sure all dependencies are installed and the virtual environment is activated:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**2. Automated Orchestration (Recommended)**
The easiest way to run the entire data collection, training, and evaluation loop is by using the orchestrator. This script will automatically loop through the configured applications, handle data collection, dynamically adjust hyperparameter tuning, and evaluate results over multiple iterations:
```bash
python3 orchestrator.py
```
> Note: Running the orchestrator might take hours depending on how many iterations and `collect_runs` are configured, since it simulates actual cold starts. 

**3. Manual Pipeline Execution**
If you want to run the machine learning pipeline stages manually for a single application, make sure the `config/config.yaml` is set for your application, and execute the stages sequentially using `main.py`:
```bash
python3 main.py collect
python3 main.py process
python3 main.py train
python3 main.py evaluate
```

---

## Switching to a New Application

You can easily configure the prefetcher to track and optimize any new application you'd like (e.g. VLC Media Player, Spotify, etc).

**Using the Orchestrator**
Open `orchestrator.py` and modify the `apps` list at the very top of the script. 
Add a new dictionary with your custom identifier `app_name` and the actual terminal executable command `target_app`:
```python
apps = [
    {"app_name": "chrome", "target_app": "google-chrome"},
    {"app_name": "gimp", "target_app": "gimp"},
    {"app_name": "firefox", "target_app": "firefox"},
    {"app_name": "vlc", "target_app": "vlc"} # <-- Add your new application here
]
```
When you run `python3 orchestrator.py`, it will automatically inject this into the configuration and run full optimizations for it.

**Using the Manual Pipeline**
If you are running the `main.py` pipeline manually, open `config/config.yaml` and directly modify the `system` configuration keys:
```yaml
system:
  app_name: vlc       # Your custom identifier (used for saving logs and models)
  target_app: vlc     # The system command used to launch the app
  collect_runs: 10
```
Then run the pipeline stages via `main.py`.
