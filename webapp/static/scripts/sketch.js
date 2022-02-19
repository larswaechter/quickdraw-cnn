const WIDTH = 500;
const HEIGHT = 500;
const STROKE_WEIGHT = 3;
const CROP_PADDING = REPOS_PADDING = 2

let model;
let pieChart;
let oldMouseX;
let oldMouseY;
let clicked = false;

let strokePixels = [[], []]
let imageStrokes = [];

function setup() {
    createCanvas(WIDTH, HEIGHT);
    strokeWeight(STROKE_WEIGHT);
    stroke("black");
    background("#FFFFFF");
}

function mouseDown() {
    if (
        mouseX <= WIDTH &&
        mouseX >= 0 &&
        mouseY <= HEIGHT &&
        mouseY >= 0) {
    }
    // console.log("Mouse down")
    clicked = true;
    oldMouseX = mouseX
    oldMouseY = mouseY
}

function mouseMoved() {
    if (
        clicked &&
        mouseX <= WIDTH &&
        mouseX >= 0 &&
        mouseY <= HEIGHT &&
        mouseY >= 0
    ) {
        strokePixels[0].push(Math.floor(mouseX))
        strokePixels[1].push(Math.floor(mouseY))

        line(mouseX, mouseY, oldMouseX, oldMouseY);

        oldMouseX = mouseX
        oldMouseY = mouseY
    }
}

function mouseReleased() {
    if (strokePixels) {
        imageStrokes.push(strokePixels);
        strokePixels = [[], []]
    }
    clicked = false;
}

const loadModel = async () => {
    console.log('Model loading...')

    model = await tflite.loadTFLiteModel("./models/model.tflite");
    model.predict(tf.zeros([1, 28, 28, 1])); // warmup

    console.log(`Model loaded! (${LABELS.length} classes)`);
}

const preprocess = async (cb) => {
    const {min, max} = getCroppedCanvasBox()

    // Resize to 28x28 pixel & crop
    const imageBlob = await fetch('/transform', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        redirect: 'follow',
        referrerPolicy: 'no-referrer',
        body: JSON.stringify({
            strokes: imageStrokes,
            box: [min.x, min.y, max.x, max.y]
        })
    }).then(response => response.blob())

    const img = new Image(28, 28);
    img.src = URL.createObjectURL(imageBlob);

    img.onload = () => {
        const tensor = tf.tidy(() => tf.browser.fromPixels(img, 1).toFloat().expandDims(0))
        cb(tensor)
    }
};

const drawPie = (top3) => {
    const probs = []
    const labels = []

    for (const pred of top3) {
        const prop = +pred.probability.toPrecision(2)
        probs.push(prop)
        labels.push(`${pred.className} (${prop})`)
    }

    const others = +(1 - probs.reduce((prev, prob) => prev + prob, 0)).toPrecision(2)
    probs.push(others)
    labels.push(`Others (${others})`)

    if (pieChart) pieChart.destroy()

    const ctx = document.getElementById("predictions").getContext("2d");
    pieChart = new Chart(ctx, {
        type: 'pie',
        options: {
            plugins: {
                legend: {
                    position: 'bottom'
                },
                title: {
                    display: true,
                    text: 'Top 3 Predictions'
                }
            }
        },
        data: {
            labels,
            datasets: [
                {
                    label: 'Top 3 predictions',
                    data: probs,
                    backgroundColor: [
                        'rgb(255, 99, 132)',
                        'rgb(54, 162, 235)',
                        'rgb(255, 205, 86)',
                        'rgb(97,96,96)'
                    ],
                }
            ]
        }
    })
}

const getMinimumCoordinates = () => {
    let min_x = Number.MAX_SAFE_INTEGER
    let min_y = Number.MAX_SAFE_INTEGER

    for (const stroke of imageStrokes) {
        for (let i = 0; i < stroke[0].length; i++) {
            min_x = Math.min(min_x, stroke[0][i])
            min_y = Math.min(min_y, stroke[1][i])
        }
    }

    return [Math.max(0, min_x), Math.max(0, min_y)]
};

const getCroppedCanvasBox = () => {
    repositionImage()

    const coords_x = [];
    const coords_y = [];

    for (const stroke of imageStrokes) {
        for (let i = 0; i < stroke[0].length; i++) {
            coords_x.push(stroke[0][i])
            coords_y.push(stroke[1][i])
        }
    }

    const x_min = Math.min(...coords_x)
    const x_max = Math.max(...coords_x)
    const y_min = Math.min(...coords_y)
    const y_max = Math.max(...coords_y)

    // New width & height of cropped image
    const width = Math.max(...coords_x) - Math.min(...coords_x)
    const height = Math.max(...coords_y) - Math.min(...coords_y)

    let coords_max = {}
    const coords_min = {
        x: Math.max(0, x_min - CROP_PADDING), // Link Kante anlegen
        y: Math.max(0, y_min - CROP_PADDING), // Obere Kante anlegen
    };

    if (width > height) // Linke + Rechte Kante als Abgrenzung
        coords_max = {
            x: Math.min(WIDTH, x_max + CROP_PADDING), // Rechte Kante anlegen
            y: Math.max(0, y_min + CROP_PADDING) + width, // Untere Kante anlegen
        }
    else // Obere + Untere Kante als Abgrenzung
        coords_max = {
            x: Math.max(0, x_min + CROP_PADDING) + height, // Rechte Kante anlegen
            y: Math.min(HEIGHT, y_max + CROP_PADDING), // Untere Kante anlegen
        };

    return {
        min: coords_min,
        max: coords_max,
    };
};

// Reposition image to top left corner
const repositionImage = () => {
    const [min_x, min_y] = getMinimumCoordinates();
    for (const stroke of imageStrokes) {
        for (let i = 0; i < stroke[0].length; i++) {
            stroke[0][i] = stroke[0][i] - min_x + REPOS_PADDING
            stroke[1][i] = stroke[1][i] - min_y + REPOS_PADDING
        }
    }
}

const predict = async () => {
    if (!imageStrokes.length) return;

    preprocess(tensor => {
        const predictions = model.predict(tensor).dataSync();

        if (!LABELS.length)
            throw new Error("No labels found!")

        const top3 = Array.from(predictions)
            .map((p, i) => ({
                probability: p,
                className: LABELS[i],
                index: i
            }))
            .sort((a, b) => b.probability - a.probability)
            .slice(0, 3);

        drawPie(top3)
    })
};

const clearCanvas = () => {
    clear();
    if (pieChart) pieChart.destroy()
    background("#FFFFFF");
    imageStrokes = [];
    strokePixels = [[], []]
}

window.onload = () => {
    const $submit = document.getElementById("predict");
    const $clear = document.getElementById("clear");
    const $canvas = document.getElementById("defaultCanvas0");

    loadModel();
    $canvas.addEventListener("mousedown", (e) => mouseDown(e));
    $canvas.addEventListener("mousemove", (e) => mouseMoved(e));

    $submit.addEventListener("click", () => predict($canvas));
    $clear.addEventListener("click", clearCanvas);
};
