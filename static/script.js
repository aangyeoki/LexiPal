const generateSpeech = () => {
    const inputText = document.getElementById("inputText").value;
    const audioPlayer = document.getElementById("audioPlayer");

    const myHeaders = new Headers();
    myHeaders.append("Content-Type", "application/json");

    const requestOptions = {
        method: "POST",
        headers: myHeaders,
        body: JSON.stringify({"text": inputText}),
        redirect: "follow"
    };

    fetch("/generate_speech", requestOptions)
        .then((response) => response.json())
        .then((result) => {
            console.log(result);
            audioPlayer.src = result.audio_file
            document.getElementById('emotionDetect').innerText = result.emotion
        })
        .catch((error) => console.error(error))
}