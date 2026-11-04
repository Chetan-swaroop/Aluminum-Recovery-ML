
"""
ML-Optimized Aluminium Recovery for Al-Air Batteries
Main Application Entry Point

Stage 1: YOLOv11m Waste Detection
Stage 2: Literature-Based Composition Profiling
Stage 3: Purification Route and Cost Prediction
Stage 4: Al-Air Battery Performance Prediction
"""

import os
import sys
import json
import logging
import argparse
import joblib

from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

YOLO_MODEL = MODEL_DIR / "stage1_yolo11m.pt"

COMPOSITION_FILE = DATA_DIR / "composition_ranges.csv"

PURITY_MODEL = MODEL_DIR / "stage3_purity.pkl"
COST_MODEL = MODEL_DIR / "stage3_cost.pkl"
ENERGY_MODEL = MODEL_DIR / "stage3_energy.pkl"

EFFICIENCY_MODEL = MODEL_DIR / "stage4_efficiency.pkl"
CORROSION_MODEL = MODEL_DIR / "stage4_corrosion.pkl"


# --------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# IMPORT PROJECT MODULES
# --------------------------------------------------

try:
    from src.stage1_detection import WasteDetector
    from src.stage2_profiling import CompositionProfiler
    from src.stage3_purification import PurificationModel
    from src.stage4_battery import BatteryModel
    from src.pipeline import AluminiumRecoveryPipeline

except ImportError as error:
    logger.error("Unable to import project modules: %s", error)
    sys.exit(1)


# --------------------------------------------------
# FILE VALIDATION
# --------------------------------------------------

def validate_files():

    required_files = [
        YOLO_MODEL,
        COMPOSITION_FILE,
        PURITY_MODEL,
        COST_MODEL,
        ENERGY_MODEL,
        EFFICIENCY_MODEL,
        CORROSION_MODEL
    ]

    missing_files = []

    for file_path in required_files:

        if not file_path.exists():
            missing_files.append(file_path)

    if missing_files:

        logger.error("Required project files are missing:")

        for file_path in missing_files:
            logger.error("Missing: %s", file_path)

        return False

    return True


# --------------------------------------------------
# LOAD STAGE 1 MODEL
# --------------------------------------------------

def load_stage1():

    logger.info("Loading YOLOv11m detection model...")

    detector = WasteDetector(
        str(YOLO_MODEL)
    )

    logger.info("Stage 1 model loaded successfully.")

    return detector


# --------------------------------------------------
# LOAD STAGE 2 MODEL
# --------------------------------------------------

def load_stage2():

    logger.info("Loading composition lookup database...")

    profiler = CompositionProfiler(
        str(COMPOSITION_FILE)
    )

    logger.info("Stage 2 database loaded successfully.")

    return profiler


# --------------------------------------------------
# LOAD STAGE 3 MODELS
# --------------------------------------------------

def load_stage3():

    logger.info("Loading purification prediction models...")

    models = {

        "Purity": joblib.load(PURITY_MODEL),

        "Cost": joblib.load(COST_MODEL),

        "Energy": joblib.load(ENERGY_MODEL)

    }

    purification = PurificationModel(models)

    logger.info("Stage 3 models loaded successfully.")

    return purification


# --------------------------------------------------
# LOAD STAGE 4 MODELS
# --------------------------------------------------

def load_stage4():

    logger.info("Loading battery performance models...")

    efficiency_model = joblib.load(
        EFFICIENCY_MODEL
    )

    corrosion_model = joblib.load(
        CORROSION_MODEL
    )

    battery = BatteryModel(
        efficiency_model,
        corrosion_model
    )

    logger.info("Stage 4 models loaded successfully.")

    return battery


# --------------------------------------------------
# INITIALIZE COMPLETE PIPELINE
# --------------------------------------------------

def initialize_pipeline():

    logger.info("=" * 55)
    logger.info("INITIALIZING ALUMINIUM RECOVERY PIPELINE")
    logger.info("=" * 55)

    if not validate_files():
        raise FileNotFoundError(
            "One or more required files are missing."
        )

    detector = load_stage1()

    profiler = load_stage2()

    purification = load_stage3()

    battery = load_stage4()

    pipeline = AluminiumRecoveryPipeline(
        detector=detector,
        profiler=profiler,
        purification=purification,
        battery=battery
    )

    logger.info("All four stages initialized successfully.")

    return pipeline


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

def display_results(results):

    print("\n")
    print("=" * 60)
    print("       ALUMINIUM RECOVERY - ANALYSIS RESULTS")
    print("=" * 60)

    if "error" in results:
        print("ERROR:", results["error"])
        return

    # Stage 1
    print("\n[STAGE 1] WASTE IDENTIFICATION")

    print(
        "Detected Material:",
        results["material"]
    )

    # Stage 2
    print("\n[STAGE 2] ELEMENTAL COMPOSITION")

    composition = results["composition"]

    print("-" * 45)

    for element, values in composition.items():

        print(
            f"{element:8s} : "
            f"{values['minimum']:.3f} - "
            f"{values['maximum']:.3f} wt% "
            f"(Estimated: {values['estimated']:.3f})"
        )

    # Stage 3
    print("\n[STAGE 3] PURIFICATION PREDICTION")

    print("-" * 60)

    for route, values in results["purification"].items():

        print(f"\nRoute: {route}")

        print(
            f"Predicted Purity: "
            f"{values['purity']:.3f}%"
        )

        print(
            f"Estimated Cost: "
            f"{values['cost']:.2f} USD/tonne"
        )

        print(
            f"Energy Consumption: "
            f"{values['energy']:.2f} kWh/tonne"
        )

    print("\nRecommended Route:")

    print(
        results["recommended_route"]
        if results["recommended_route"]
        else "No suitable route found."
    )

    # Stage 4
    print("\n[STAGE 4] BATTERY PERFORMANCE")

    battery_results = results["battery_performance"]

    if battery_results:

        print(
            "Discharge Efficiency:",
            f"{battery_results['discharge_efficiency']:.2f}%"
        )

        print(
            "Self-Corrosion Rate:",
            f"{battery_results['self_corrosion_rate']:.4f}",
            "mg cm-2 min-1"
        )

    else:
        print("Battery prediction skipped.")

    print("\n" + "=" * 60)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_file = OUTPUT_DIR / (
        f"analysis_{timestamp}.json"
    )

    with open(output_file, "w") as file:

        json.dump(
            results,
            file,
            indent=4
        )

    logger.info(
        "Results saved to: %s",
        output_file
    )


# --------------------------------------------------
# MAIN EXECUTION FUNCTION
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Aluminium Recovery ML Pipeline"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input waste image"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    # Validate input image
    if not image_path.exists():

        logger.error(
            "Input image does not exist: %s",
            image_path
        )

        sys.exit(1)

    try:

        # Initialize all four stages
        pipeline = initialize_pipeline()

        logger.info("Processing image: %s", image_path)

        # Execute complete pipeline
        results = pipeline.process(
            str(image_path)
        )

        # Display results
        display_results(results)

        # Save results
        save_results(results)

        logger.info(
            "Pipeline execution completed successfully."
        )

    except Exception as error:

        logger.exception(
            "Pipeline execution failed: %s",
            error
        )

        sys.exit(1)


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    main()
