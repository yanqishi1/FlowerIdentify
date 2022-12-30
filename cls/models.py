from django.db import models
from django.contrib import admin
# Create your models here.

class ImageCls(models.Model):
    id = models.AutoField(primary_key=True)
    image_path = models.CharField(max_length= 100)
    pred = models.CharField(max_length= 100)

@admin.register(ImageCls)
class ImageClsAdmin(admin.ModelAdmin):
    list_display = ("id","image_path" ,"pred")
    ordering = ("id",)