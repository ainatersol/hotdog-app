document.getElementById('image-upload').addEventListener('change', function() {
    var reader = new FileReader();
    reader.onload = function() {
      var img = document.getElementById('uploaded-image');
      img.src = reader.result;
      img.style.display = 'block';
    };
    reader.readAsDataURL(this.files[0]);
  });
  
  document.getElementById('predict-button').addEventListener('click', function() {
    var img = document.getElementById('uploaded-image');
    fetch('https://ainatersol-isithotdog.hf.space/predict', {
      method: 'POST',
      body: JSON.stringify({
        img: img.src,
      }),
    })
      .then(response => response.json())
      .then(data => {
        // Display the prediction result in the "result" div
        var resultDiv = document.getElementById('result');
        resultDiv.innerHTML = data.result; // Update this to match the structure of your API response
      });
  });
  