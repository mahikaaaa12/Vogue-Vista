// 1. Package the order (The image)
const formData = new FormData();
formData.append("file", selectedFile); 

// 2. The Waiter carries the order to the kitchen
const response = await fetch("http://127.0.0.1:8000/api/v1/analyze-color", {
    method: "POST",
    body: formData,
});

// 3. The Waiter brings the food (JSON data) back to the table!
const data = await response.json();

// 4. Put the food on the table (Update the HTML)
document.getElementById('resSkinTone').textContent = data.skin_tone;
document.getElementById('resSeason').textContent = data.palette_category;