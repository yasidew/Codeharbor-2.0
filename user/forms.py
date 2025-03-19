from django import forms
from .models import UserProfile

class ProfileUpdateForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = ["avatar", "bio"]

    def clean_avatar(self):
        avatar = self.cleaned_data.get("avatar")
        if avatar:
            allowed_types = ["image/jpeg", "image/png", "image/jpg"]
            if avatar.content_type not in allowed_types:
                raise forms.ValidationError("Only JPEG, PNG, and JPG files are allowed.")
        return avatar
