import os


def load_huggingface_model(model_class, model_name, local_file, **kwargs):
    if os.path.exists(local_file):
        print(f"Loading {model_class} from local file {local_file}")
        model = model_class.from_pretrained(local_file)
    else:
        print(f"Loading {model_class} from huggingface model {model_name}")
        model = model_class.from_pretrained(model_name)
        model.save_pretrained(local_file)
        print(f"Saved {model_class} to local file {local_file}")
    if kwargs.get("return_tokenizer", None) == True:
        return model.tokenizer
    elif kwargs.get("return_feature_extractor", None) == True:
        return model.feature_extractor
    elif kwargs.get("return_vision_model", None) == True:
        return model.vision_model
    return model
