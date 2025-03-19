# Configuration variables
# LLM_URL = "http://192.168.1.165:1234/v1/chat/completions"
LLM_URL = "http://localhost:1234/v1/chat/completions"
HEADERS = {"Content-Type": "application/json"}

# Sample HTML content for testing
SAMPLE_HTML_HEALTHY = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WCAG 2.1 Compliance Example</title>
</head>
<body>

    <!-- Non-text Content (1.1.1) -->
    <h1>Image Example with Text Alternative (1.1.1)</h1>
    <img src="cat.jpg" alt="A cute cat sitting on a couch" />

    <h2>Chart Example with Text Alternative (1.1.1)</h2>
    <img src="chart.png" alt="Bar chart showing monthly sales" />

    <!-- Audio-only and Video-only (Prerecorded) (1.2.1) -->
    <h1>Audio-only Content with Transcript (1.2.1)</h1>
    <audio controls>
        <source src="audio.mp3" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    <p>Transcript: This is an audio file that discusses the importance of web accessibility.</p>

    <h2>Video-only Content with Audio Description (1.2.1)</h2>
    <video controls>
        <source src="video.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <audio controls>
        <source src="video_description.mp3" type="audio/mpeg">
        Your browser does not support the audio description.
    </audio>

    <!-- Captions (Prerecorded) (1.2.2) -->
    <h1>Video with Captions (1.2.2)</h1>
    <video controls>
        <source src="video_with_captions.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <track src="captions_en.vtt" kind="captions" srclang="en" label="English" />

    <!-- Audio Description or Media Alternative (Prerecorded) (1.2.3) -->
    <h1>Video with Audio Description (1.2.3)</h1>
    <video controls>
        <source src="video_with_audio_description.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <audio controls>
        <source src="audio_description.mp3" type="audio/mpeg">
        Your browser does not support the audio description.
    </audio>

    <!-- Captions (Live) (1.2.4) -->
    <h1>Live Stream with Captions (1.2.4)</h1>
    <video controls>
        <source src="live_stream.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <track src="live_captions.vtt" kind="captions" srclang="en" label="English" />

    <!-- Audio Description (Prerecorded) (1.2.5) -->
    <h1>Video with Audio Description (1.2.5)</h1>
    <video controls>
        <source src="video_with_audio_description.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <audio controls>
        <source src="audio_description.mp3" type="audio/mpeg">
        Your browser does not support the audio description.
    </audio>

    <!-- Sign Language (Prerecorded) (1.2.6) -->
    <h1>Video with Sign Language Interpretation (1.2.6)</h1>
    <video controls>
        <source src="video_with_sign_language.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <video controls>
        <source src="sign_language_interpreter.mp4" type="video/mp4">
        Your browser does not support the sign language interpreter.
    </video>

    <!-- Extended Audio Description (Prerecorded) (1.2.7) -->
    <h1>Extended Audio Description for Video (1.2.7)</h1>
    <video controls>
        <source src="extended_audio_description_video.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <audio controls>
        <source src="extended_audio_description.mp3" type="audio/mpeg">
        Your browser does not support the extended audio description.
    </audio>

    <!-- Media Alternative (Prerecorded) (1.2.8) -->
    <h1>Media Alternative for Time-Based Media (1.2.8)</h1>
    <video controls>
        <source src="video_with_media_alternative.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    <a href="alternative_media_link.pdf">Download media alternative</a>

    <!-- Audio-only (Live) (1.2.9) -->
    <h1>Live Audio-only Content with Transcript (1.2.9)</h1>
    <audio controls>
        <source src="live_audio.mp3" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    <p>Transcript: This is a live audio-only broadcast of an important conference.</p>

</body>
</html>"""

SAMPLE_HTML_BAD = """<html>
    <body>
    <h1>heres a picture of a cat</h1>
    <img src="cat.jpg">
    <h2>some chart i guess</h2>
    <img src="chart.png">

    <h1>some audio stuff</h1>
    <audio src="audio.mp3"></audio>
    <p>this is audio about something who cares</p>

    <h2>video or whatever</h2>
    <video src="video.mp4"></video>
    <audio src="video_description.mp3"></audio>

    <h1>video with stuff</h1>
    <video src="video_with_captions.mp4"></video>

    <h1>another video</h1>
    <video src="video_with_audio_description.mp4"></video>
    <audio src="audio_description.mp3"></audio>

    <h1>live video</h1>
    <video src="live_stream.mp4"></video>

    <h1>video again</h1>
    <video src="video_with_audio_description.mp4"></video>
    <audio src="audio_description.mp3"></audio>

    <h1>sign language video</h1>
    <video src="video_with_sign_language.mp4"></video>
    <video src="sign_language_interpreter.mp4"></video>

    <h1>long audio video</h1>
    <video src="extended_audio_description_video.mp4"></video>
    <audio src="extended_audio_description.mp3"></audio>

    <h1>video with a link</h1>
    <video src="video_with_media_alternative.mp4"></video>
    <a href="alternative_media_link.pdf">click here lol</a>

    <h1>live audio</h1>
    <audio src="live_audio.mp3"></audio>
    <p>some conference thing idk</p>

    </body>
    </html>"""