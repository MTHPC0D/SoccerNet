from ultralytics import YOLO

def load_model(model_id: str):
    model = YOLO(model_id) 
    return model

def train(model, data_path: str, epochs: int=50, imgsz: int=416, batch:int  = 8, verbose: bool=True):
    results = model.train(data=data_path)
    return results


