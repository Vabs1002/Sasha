import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import yaml
import logging
from typing import Dict, Any, Tuple, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "role_detector_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

def load_or_train_model(signals_dict: Dict[str, Any]) -> Tuple[LogisticRegression, TfidfVectorizer]:
    """
    Load existing model and vectorizer, or train new ones from signals dictionary.

    Args:
        signals_dict: Dictionary containing role signals for training

    Returns:
        Tuple of (classifier, vectorizer)

    Raises:
        Exception: For training or loading errors
    """
    # If model files exist, load them (regardless of signals_dict content)
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            logger.info(f"Loading existing model from {MODEL_PATH} and {VECTORIZER_PATH}")
            with open(MODEL_PATH, 'rb') as f:
                clf = pickle.load(f)
            with open(VECTORIZER_PATH, 'rb') as f:
                vec = pickle.load(f)
            return clf, vec
        except Exception as e:
            logger.warning(f"Failed to load existing model: {str(e)}. Will retrain if possible.")

    # If we reach here, either models don't exist or loading failed
    # Handle empty signals_dict by creating a minimal default model
    if not signals_dict:
        logger.info("No signals provided and no existing model found. Creating minimal default model.")
        # Create a minimal default model with a single generic role
        texts = ["general experience skills projects"]
        labels = ["generic_role"]
    else:
        # Train on a small synthetic dataset derived from signals_dict
        logger.info("Training new role detection model from signals")

        # Build training data: for each role, create pseudo-resume snippets from its signals
        texts = []
        labels = []
        for role, config in signals_dict.items():
            if not isinstance(config, dict):
                logger.warning(f"Invalid config for role {role}: {config}")
                continue

            weight = config.get('weight', 1.0)
            # Create a few example strings per role
            skills = " ".join(config.get('skills', []))
            projects = " ".join(config.get('projects', []))

            if not skills and not projects:
                logger.warning(f"No skills or projects found for role {role}")
                continue

            # Combine multiple times to increase weight
            for _ in range(int(weight * 2)):  # weight influences number of examples
                texts.append(skills + " " + projects)
                labels.append(role)

        if not texts or not labels:
            logger.warning("No valid training data could be generated from signals, creating minimal default model")
            # Fallback to minimal default model
            texts = ["general experience skills projects"]
            labels = ["generic_role"]

    logger.debug(f"Generated {len(texts)} training examples for {len(set(labels))} roles")

    # Vectorize and train
    try:
        vec = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=500)
        X = vec.fit_transform(texts)
        clf = LogisticRegression(max_iter=1000, multi_class='ovr')
        clf.fit(X, labels)

        # Save model
        logger.info(f"Saving trained model to {MODEL_PATH} and {VECTORIZER_PATH}")
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(clf, f)
        with open(VECTORIZER_PATH, 'wb') as f:
            pickle.dump(vec, f)

        return clf, vec
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def detect_role(resume_text: str, signals_dict: Dict[str, Any]) -> Dict[str, Union[str, float, Dict[str, float]]]:
    """
    Detect the most likely role from resume text using trained model.

    Args:
        resume_text: The text content of the resume
        signals_dict: Dictionary containing role signals

    Returns:
        Dictionary with role detection results:
        - role: Predicted role
        - confidence: Confidence score (0-1)
        - scores: Dictionary of all role scores

    Raises:
        ValueError: If inputs are invalid
        Exception: For detection errors
    """
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text cannot be empty")

    if not signals_dict:
        raise ValueError("Signals dictionary cannot be empty")

    try:
        clf, vec = load_or_train_model(signals_dict)
        X = vec.transform([resume_text])
        probs = clf.predict_proba(X)[0]
        classes = clf.classes_

        # Get class with highest probability
        idx = np.argmax(probs)
        role = classes[idx]
        confidence = float(probs[idx])

        # Also return scores dict for transparency
        scores = {cls: float(prob) for cls, prob in zip(classes, probs)}

        result = {
            'role': role,
            'confidence': round(confidence, 3),
            'scores': {k: round(v, 3) for k, v in scores.items()}
        }

        logger.info(f"Detected role: {role} with confidence {confidence:.3f}")
        return result

    except Exception as e:
        logger.error(f"Error during role detection: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python signal_detector.py <role_signals.yaml> <resume_text>")
        sys.exit(1)
    signals = yaml.safe_load(open(sys.argv[1], 'r'))
    text = sys.argv[2]
    result = detect_role(text, signals)
    import json
    print(json.dumps(result, indent=2))