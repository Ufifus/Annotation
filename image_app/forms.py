from django import forms

from .models import ProjectModel, ImageModel


class ProjForm(forms.Form):
    """Форма для добавления проектов"""
