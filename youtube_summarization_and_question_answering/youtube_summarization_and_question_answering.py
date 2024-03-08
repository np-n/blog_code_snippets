import os
import io
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi

# configure gemini api
load_dotenv()
GEMINI_API_KEY = os.getenv(key="GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


# initialize gemini pro model
def initialize_model(model_name="gemini-pro"):
    model = genai.GenerativeModel(model_name)
    return model


def get_response(model, prompt):
    """
    Method to get track the applicant (ATS system) using job requirement and resume
    :param model:
    :param prompt:
    :return:
    """
    response = model.generate_content(prompt)
    return response.text


def get_video_id(url):
    """
    Method go get video_id from URL
    :param url:
    :return:
    """
    video_id = url.split("=")[1]
    if "&" in video_id:
        video_id = video_id.split("&")[0]

    return video_id


def get_video_transcripts(video_id):
    """
    Method to transcribe the YouTube video
    :param video_id:
    :return:
    """
    try:
        transcription_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcription = " ".join([transcript["text"] for transcript in transcription_list])
        return transcription

    except Exception as e:
        raise e


def main():
    # Sidebar content
    with st.sidebar:
        st.subheader("Choose your application!")
        st.markdown("<br>", unsafe_allow_html=True)
        option = st.selectbox(
            'Which application would you like to run?',
            ('YouTube video summarizer', 'Question Answering in YouTube video')),

    if option[0] == "YouTube video summarizer":
        st.title("YouTube video summarizer")
        st.markdown("<br>", unsafe_allow_html=True)
        youtube_url = st.text_input("Enter youtube video link:")
        if youtube_url:
            video_id = get_video_id(youtube_url)
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
        submit = st.button("submit")

        # define prompt here
        model_behavior = """ You are expert in summarization of youtube video from transcription of video.
                    So, input is transcription and output will be the summary of the given video including all 
                    the important information. Please break down the information in multiple paragraph if it becomes
                    more clear and concise.Please give relevant topic for the summary.
                    Please try to make the summary in below 1000 words. Please don't add extra information that doesn't
                    make sense but fix typos and return `Couldn't generate summary for the given video` if transcription is meaningless or empty.
                    This is the transcriptions for the video.
                """

        if submit:
            transcriptions = get_video_transcripts(video_id)


            # initialize the gemini-pro model
            gemini_model = initialize_model(model_name="gemini-pro")
            final_prompt = model_behavior + "\n\n" + transcriptions
            summary = get_response(model=gemini_model, prompt=final_prompt)
            st.write(summary)

    if option[0] == "Question Answering in YouTube video":
        st.title("Question Answering in YouTube video")
        st.markdown("<br>", unsafe_allow_html=True)
        youtube_url = st.text_input("Enter youtube video link:")
        if youtube_url:
            video_id = get_video_id(youtube_url)
            st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

        user_prompt = st.text_area("Your Prompt on above video", key="user_prompt")
        submit = st.button("submit")

        model_behavior = """ You are expert in summarization of youtube videos from transcription of videos.
            So, input is transcriptions of videos along with prompt which have the user query. Please make sure that you have
             understand all the information present in the video from transcription and respond user query. 
             Please don't add extra information that doesn't make sense but fix typos and return `Couldn't transcribe the video` 
             if transcription of video is empty otherwise respond accordingly!.
        """
        if user_prompt or submit:
            # transcribe the video
            video_transcriptions = get_video_transcripts(video_id)
            # initialize the gemini-pro model
            gemini_model = initialize_model(model_name="gemini-pro")
            # add transcription and prompt to main prompt
            model_behavior = model_behavior + f"\nvideo transcription: {video_transcriptions} \nprompt: {user_prompt}"

            response = get_response(model=gemini_model, prompt=model_behavior)
            st.write(response)


if __name__ == "__main__":
    main()
