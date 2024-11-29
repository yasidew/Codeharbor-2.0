from django.contrib import admin
from .models import Drink
from .models import Challenge

admin.site.register(Drink),
admin.site.register(Challenge)