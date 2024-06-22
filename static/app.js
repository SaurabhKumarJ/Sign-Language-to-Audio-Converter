let model;
const webcamElement = document.getElementById('webcam');
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const predictionElement = document.getElementById('prediction');
let isPredicting = false;

async function loadModel() {
    try {
        console.log('Loading model...');
        model = await tflite.loadTFLiteModel('/static/model.tflite');
        console.log('Model loaded successfully.');
    } catch (error) {
        console.error('Error loading model:', error);
    }
}


async function setupWebcam() {
    return new Promise((resolve, reject) => {
        const navigatorAny = navigator;
        navigator.getUserMedia = navigator.getUserMedia ||
            navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
            navigatorAny.msGetUserMedia;
        if (navigator.getUserMedia) {
            navigator.getUserMedia(
                { video: true },
                stream => {
                    webcamElement.srcObject = stream;
                    webcamElement.addEventListener('loadeddata', () => {
                        console.log('Webcam initialized.');
                        resolve();
                    }, false);
                },
                error => {
                    console.error('Error accessing webcam:', error);
                    reject(error);
                }
            );
        } else {
            reject(new Error('getUserMedia not supported in this browser.'));
        }
    });
}

async function predict() {
    while (isPredicting) {
        if (model && webcamElement.readyState === 4) { // Check if model and video are ready
            const tensor = tf.browser.fromPixels(webcamElement)
                .resizeBilinear([224, 224])
                .expandDims(0)
                .toFloat();
            try {
                const prediction = await model.predict(tensor);
                predictionElement.innerText = `Prediction: ${prediction}`;
            } catch (error) {
                console.error('Error during prediction:', error);
            } finally {
                tensor.dispose(); // Clean up tensors to avoid memory leaks
            }
        }
        await tf.nextFrame();
    }
}


startButton.addEventListener('click', () => {
    isPredicting = true;
    predict();
});

stopButton.addEventListener('click', () => {
    isPredicting = false;
});

(async () => {
    await loadModel();
    await setupWebcam();
})();
