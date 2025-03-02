from django import forms
from .models import Guideline, DesignPatternResource


class GuidelineForm(forms.ModelForm):
    class Meta:
        model = Guideline
        fields = ['company_name', 'pattern', 'rule', 'company_logo']


class DesignPatternResourceForm(forms.ModelForm):
    class Meta:
        model = DesignPatternResource
        fields = ['pattern_name', 'description', 'link', 'category']
        widgets = {
            'pattern_name': forms.Select(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
            'link': forms.URLInput(attrs={'class': 'form-control'}),
            'category': forms.Select(attrs={'class': 'form-control'}),
        }
