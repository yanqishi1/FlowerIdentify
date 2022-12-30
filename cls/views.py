
from django.shortcuts import render,HttpResponse
# Create your views here.
from models.Detector import Detector
import datetime
from cls.models import ImageCls
import json


detector = Detector()

def Home(request):
    '''功能测试页面'''
    return render(request, "Home.html")


def detect(request):
    pred = ''
    message = ''

    if request.method == 'POST':
        image = request.FILES.get("file", None)
        if not image:
            pred = -1
            message = 'Predict Error'
        else:
            # 使用时间戳命名图片，防止文件名过长或重复
            # 获取文件类型
            file_type = str(image.name).split(".")[1]
            # 时间戳
            timestamp = str(datetime.datetime.now()).replace(":","-")+ "." + file_type
            path = "./upload/" + timestamp

            dest = open(path, "wb+")
            for chunk in image.chunks():
                dest.write(chunk)

            pred, message = detector.predict(path)
            ImageCls.objects.create(image_path=timestamp, pred=pred,time=datetime.datetime.now())
        return pred,message



def predict(request):
    pred,message = detect(request)
    return HttpResponse(json.dumps({'pred': pred, 'message': message}))

def predict_with_page(request):
    pred, message = detect(request)
    return render(request,
                  "result.html",
                  {'result': pred, 'message': message}
                  )