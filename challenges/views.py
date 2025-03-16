from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated

from .models import Challenges
from datetime import datetime

from django.shortcuts import render

def challenges_page(request):
    return render(request, 'current_challenge.html')

def challenge_details_page(request):
    return render(request, 'selectedChallenge.html')


@api_view(['GET'])
@permission_classes([IsAuthenticated])
#@permission_classes([AllowAny])
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
@permission_classes([IsAuthenticated])
#@permission_classes([AllowAny])
def get_challenges_for_current_month(request):
    current_month = datetime.now().month
    challenges = Challenges.objects.filter(month=current_month)

    if request.headers.get('Accept') == 'application/json':  # Check if it's an API request
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
        return Response(
            {"error": f"No challenges found for the current month ({current_month})."},
            status=status.HTTP_404_NOT_FOUND,
        )

    # If it's a regular web request, render the HTML template
    return render(request, 'current_challenge.html', {'challenges': challenges})


@api_view(['GET'])
@permission_classes([IsAuthenticated])
#@permission_classes([AllowAny])
def get_challenges_by_month(request, month):
    try:
        month = int(month)
        if month < 1 or month > 12:
            raise ValueError("Invalid month value")
    except ValueError:
        return Response(
            {"error": "Invalid month. Please provide a value between 1 and 12."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    challenges = Challenges.objects.filter(month=month)
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

    return Response(
        {"error": f"No challenges found for month {month}."},
        status=status.HTTP_404_NOT_FOUND,
    )


@api_view(['POST'])
# @permission_classes([IsAuthenticated])
def create_challenge(request):
    try:
        data = request.data
        category = data.get("category")
        template = data.get("template")
        difficulty = data.get("difficulty")
        month = data.get("month")

        # Validate inputs
        if not category or not template or not difficulty or not month:
            return Response(
                {"error": "All fields (category, template, difficulty, month) are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Validate month
        try:
            month = int(month)
            if month < 1 or month > 12:
                raise ValueError("Invalid month")
        except ValueError:
            return Response(
                {"error": "Invalid month. Please provide a value between 1 and 12."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Create the challenge
        challenge = Challenges.objects.create(
            category=category,
            template=template,
            difficulty=difficulty,
            month=month,
        )
        return Response(
            {
                "message": "Challenge created successfully.",
                "challenge": {
                    "id": challenge.id,
                    "category": challenge.category,
                    "template": challenge.template,
                    "difficulty": challenge.difficulty,
                    "month": challenge.month,
                    "created_at": challenge.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                },
            },
            status=status.HTTP_201_CREATED,
        )
    except Exception as e:
        return Response(
            {"error": "An error occurred while creating the challenge.", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

@api_view(['GET'])
# @permission_classes([IsAuthenticated])
def get_challenge_by_id(request, challenge_id):
    print(request.headers.get('Authorization'))
    """
    Fetch details of a specific challenge by its ID.
    """
    try:
        challenge = get_object_or_404(Challenges, id=challenge_id)
        challenge_data = {
            "id": challenge.id,
            "category": challenge.category,
            "template": challenge.template,
            "difficulty": challenge.difficulty,
            "month": challenge.month,
            "created_at": challenge.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            # "description": challenge.description,  # Remove this line
        }
        return Response(challenge_data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response(
            {"error": "An error occurred while fetching the challenge details.", "details": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

