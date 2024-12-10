from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

# Create your views here.
from django.contrib.auth.models import User
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.tokens import RefreshToken, OutstandingToken
from rest_framework.exceptions import NotAuthenticated
from django.core.exceptions import ValidationError


class UserAPI(APIView):
    authentication_classes = [JWTAuthentication]
    permission_classes = [AllowAny]

    def get(self, request):
        user = request.user
        if not user.is_authenticated:
            raise NotAuthenticated()
        return Response({
            "username": user.username,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
        })

    def post(self, request):
        username = request.data.get("username", "")
        password = request.data.get("password", "")

        if username and password:
            if User.objects.filter(username=username).exists():
                return Response({
                    "error": "A user with that username already exists."
                }, status=400)

            # Use create_user to hash the password
            user = User.objects.create_user(username=username, password=password)
            refresh = RefreshToken.for_user(user)
            return Response({
                "refresh": str(refresh),
                'access': str(refresh.access_token),
            }, status=201)

        return Response({
            "error": "Both username and password must be provided."
        }, status=400)

    def put(self, request):
        user = request.user  # Get the authenticated user
        data = request.data

        # Extract the new details from the request
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        email = data.get('email', '').strip()

        # Validate the email if it's being updated
        if email and User.objects.filter(email=email).exclude(username=user.username).exists():
            return Response({
                "error": "This email is already associated with another account."
            }, status=400)

        # Update the user's profile information
        user.first_name = first_name if first_name else user.first_name
        user.last_name = last_name if last_name else user.last_name
        user.email = email if email else user.email

        try:
            user.save()
            return Response({
                "message": "Profile updated successfully.",
                "user": {
                    "username": user.username,
                    "first_name": user.first_name,
                    "last_name": user.last_name,
                    "email": user.email
                }
            }, status=200)
        except ValidationError as e:
            return Response({"error": str(e)}, status=400)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout(request):
    refresh_token = request.data.get('refresh_token')

    if refresh_token:
        try:
            token = RefreshToken(refresh_token)
            token.blacklist()
            return Response({'message': 'Successfully logged out.'}, status=200)
        except Exception as e:
            return Response({'error': 'Invalid token.'}, status=400)
    else:
        return Response({'error': 'Refresh token is required.'}, status=400)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_all(request):
    user = request.user
    tokens = OutstandingToken.objects.filter(user=user)
    for token in tokens:
        token.blacklist()
    return Response({'message': 'Successfully logged out from all sessions.'}, status=200)

