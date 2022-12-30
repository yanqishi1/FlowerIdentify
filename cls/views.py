
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
            name = 'Not Found'
            message = 'Predict Error'
            url = ''
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

            name,message,url = detector.predict(path)
            ImageCls.objects.create(image_path=timestamp, pred=name)
        return name,message,url

def get_img(request):
    image_name = request.GET['name']

    if image_name is None or image_name=='':
        request.session['current_image'] = None
        image_data = open("./static/default.jpg", "rb").read()
    else:
        image_path = "./static/flowers/" + image_name
        request.session['current_image'] = image_name
        image_data = open(image_path, "rb").read()

    return HttpResponse(image_data, content_type="image/jpg")


def predict(request):
    name, desc,img_url = detect(request)
    return HttpResponse(json.dumps({'name': name, 'desc': desc,'img_url':img_url}))

def predict_with_page(request):
    name, desc,img_url = detect(request)
    return render(request,
                  "result.html",
                  {'name': name, 'desc': desc,'img_url':img_url}
                  )