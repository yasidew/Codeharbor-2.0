from django.shortcuts import render

# Create your views here.
# challenges/views.py
from django.http import JsonResponse
from datetime import datetime

from rest_framework.permissions import AllowAny

from .models import Challenges
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from .accessibility_checker import check_accessibility

@api_view(['GET'])
@permission_classes([AllowAny])
def get_all_challenges(request):
    challenges = Challenges.objects.all()
    challenges_data = [
        {
            "id": challenge.id,
            "category": challenge.category,
            "template": challenge.template,
            "difficulty": challenge.difficulty,
            "month": challenge.month,
            "created_at": challenge.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }
        for challenge in challenges
    ]
    return Response({"challenges": challenges_data}, status=status.HTTP_200_OK)

@api_view(['GET'])
def get_challenges_for_current_month(request):
    current_month = datetime.now().month
    challenges = Challenges.objects.filter(month=current_month)
    if challenges.exists():
        challenges_data = [
            {
                "id": challenge.id,
                "category": challenge.category,
                "template": challenge.template,
                "difficulty": challenge.difficulty,
                "month": challenge.month,
                "created_at": challenge.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for challenge in challenges
        ]
        return Response({"challenges": challenges_data}, status=status.HTTP_200_OK)
    return Response({"error": f"No challenges found for the current month ({current_month})."},
                    status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
def get_challenges_by_month(request, month):
    try:
        # Ensure the month is a valid integer between 1 and 12
        month = int(month)
        if month < 1 or month > 12:
            raise ValueError("Invalid month value")
    except ValueError:
        return Response({"error": "Invalid month. Please provide a value between 1 and 12."},
                        status=status.HTTP_400_BAD_REQUEST)

    # Filter challenges for the given month
    challenges = Challenges.objects.filter(month=month)

    if challenges.exists():
        # Serialize the challenges
        challenges_data = [
            {
                "id": challenge.id,
                "category": challenge.category,
                "template": challenge.template,
                "difficulty": challenge.difficulty,
                "month": challenge.month,
                "created_at": challenge.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for challenge in challenges
        ]
        return Response({"challenges": challenges_data}, status=status.HTTP_200_OK)

    # Return an error response if no challenges are found
    return Response({"error": f"No challenges found for month {month}."}, status=status.HTTP_404_NOT_FOUND)

