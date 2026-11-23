# DDoS Attack Detection in Software-Defined Networks Using Lightweight Deep Learning

## Project overview
This is a defensive, classroom-ready prototype. It uses a small Multilayer Perceptron (MLP) neural network to classify **synthetic SDN flow-statistics records** as `Normal` or `DDoS-like`.

The model uses six illustrative flow features:
- Flow duration
- Packets per second
- Bytes per second
- Average packet size
- Active flow count
- SYN packet ratio

The project does not generate attack traffic, send packets, connect to a live SDN controller, or automatically block network traffic.

## Requirements
- Python 3.10 or newer
- Visual Studio Code
- Internet connection for the first installation of Python packages

## Run in VS Code

1. Extract the ZIP file.
2. Open VS Code.
3. Select **File → Open Folder** and choose `SDN_DDoS_Detection_Prototype`.
4. Open **Terminal → New Terminal**.
5. Create a virtual environment:

   Windows:
   ```bash
   py -m venv .venv
   .venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

6. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```

7. Train the model and create the dataset:
   ```bash
   python train_model.py
   ```

8. Start the dashboard:
   ```bash
   streamlit run app.py
   ```

9. Streamlit will show a local address (usually `http://localhost:8501`). Open it in your browser.

## Dashboard pages
- **Dashboard:** dataset summary, class distribution, and test metrics.
- **Test a traffic flow:** enter example flow statistics and receive a model classification.
- **Dataset:** inspect and download the generated CSV.

## Project structure
```text
SDN_DDoS_Detection_Prototype/
├── app.py
├── train_model.py
├── generate_dataset.py
├── requirements.txt
├── README.md
├── data/
│   └── sdn_demo_dataset.csv
└── models/
    └── ddos_mlp_model.joblib
```

## Important academic note
The bundled dataset is synthetic and intentionally simplified. Reported scores measure performance on a held-out split of this generated data only. They are **not** evidence of accuracy on real networks. For research-grade evaluation, replace it with a properly licensed, documented SDN flow dataset, keep train/test separation by capture or time period, and report false-positive rate and inference latency.

## Suggested future enhancement
Connect a read-only SDN telemetry source (for example, OpenFlow flow statistics from a controlled Mininet lab) and map its measured fields to the six model features. Keep detection separate from any traffic-control action.
