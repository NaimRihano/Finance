document.getElementById("predictionForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const formData = new FormData(this);
    const data = {};

    formData.forEach((value, key) => {
        data[key] = value;
    });

    fetch("/predict", {
        method: "POST",
        body: new URLSearchParams(data),
    })
    .then(response => response.json())
    .then(data => {
        const resultDiv = document.getElementById("result");
        resultDiv.innerHTML = `
            <h2>Result</h2>
            <p>Prediction: ${data.prediction}</p>
            <p>Confidence Score: ${data.confidence_score}</p>
        `;
    })
    .catch(error => console.error("Error:", error));
});
