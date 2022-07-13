from django.contrib import admin
from .models import ProjectModel, ImageModel, RoisModel

admin.site.register(ProjectModel)
admin.site.register(ImageModel)
admin.site.register(RoisModel)
