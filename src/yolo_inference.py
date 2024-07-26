from ultralytics import YOLO 

model = YOLO('/Users/mathieu/Programs/soccer-net/model/best.pt')

results = model.predict('input_videos/video1.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)