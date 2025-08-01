(function() {
  "use strict";

  /**
   * Easy selector helper function
   */
  const select = (el, all = false) => {
    el = el.trim()
    if (all) {
      return [...document.querySelectorAll(el)]
    } else {
      return document.querySelector(el)
    }
  }

  /**
   * Easy event listener function
   */
  const on = (type, el, listener, all = false) => {
    let selectEl = select(el, all)
    if (selectEl) {
      if (all) {
        selectEl.forEach(e => e.addEventListener(type, listener))
      } else {
        selectEl.addEventListener(type, listener)
      }
    }
  }

  /**
   * Easy on scroll event listener 
   */
  const onscroll = (el, listener) => {
    el.addEventListener('scroll', listener)
  }

  /**
   * Navbar links active state on scroll
   */
  let navbarlinks = select('#navbar .scrollto', true)
  const navbarlinksActive = () => {
    let position = window.scrollY + 200
    navbarlinks.forEach(navbarlink => {
      if (!navbarlink.hash) return
      let section = select(navbarlink.hash)
      if (!section) return
      if (position >= section.offsetTop && position <= (section.offsetTop + section.offsetHeight)) {
        navbarlink.classList.add('active')
      } else {
        navbarlink.classList.remove('active')
      }
    })
  }
  window.addEventListener('load', navbarlinksActive)
  onscroll(document, navbarlinksActive)

  /**
   * Scrolls to an element with header offset
   */
  const scrollto = (el) => {
    let header = select('#header')
    let offset = header.offsetHeight

    let elementPos = select(el).offsetTop
    window.scrollTo({
      top: elementPos - offset,
      behavior: 'smooth'
    })
  }

  /**
   * Toggle .header-scrolled class to #header when page is scrolled
   */
  let selectHeader = select('#header')
  let selectTopbar = select('#topbar')
  if (selectHeader) {
    const headerScrolled = () => {
      if (window.scrollY > 100) {
        selectHeader.classList.add('header-scrolled')
        if (selectTopbar) {
          selectTopbar.classList.add('topbar-scrolled')
        }
      } else {
        selectHeader.classList.remove('header-scrolled')
        if (selectTopbar) {
          selectTopbar.classList.remove('topbar-scrolled')
        }
      }
    }
    window.addEventListener('load', headerScrolled)
    onscroll(document, headerScrolled)
  }

  /**
   * Back to top button
   */
  let backtotop = select('.back-to-top')
  if (backtotop) {
    const toggleBacktotop = () => {
      if (window.scrollY > 100) {
        backtotop.classList.add('active')
      } else {
        backtotop.classList.remove('active')
      }
    }
    window.addEventListener('load', toggleBacktotop)
    onscroll(document, toggleBacktotop)
  }

 

  /**
   * Scrool with ofset on links with a class name .scrollto
   */
  on('click', '.scrollto', function(e) {
    if (select(this.hash)) {
      e.preventDefault()

      let navbar = select('#navbar')
      if (navbar.classList.contains('navbar-mobile')) {
        navbar.classList.remove('navbar-mobile')
        let navbarToggle = select('.mobile-nav-toggle')
        navbarToggle.classList.toggle('bi-list')
        navbarToggle.classList.toggle('bi-x')
      }
      scrollto(this.hash)
    }
  }, true)

  /**
   * Scroll with ofset on page load with hash links in the url
   */
  window.addEventListener('load', () => {
    if (window.location.hash) {
      if (select(window.location.hash)) {
        scrollto(window.location.hash)
      }
    }
  });

  /**
   * Preloader
   */
  let preloader = select('#preloader');
  if (preloader) {
    window.addEventListener('load', () => {
      preloader.remove()
    });
  }

  /**
   * Hero carousel indicators
   */
  let heroCarouselIndicators = select("#hero-carousel-indicators")
  let heroCarouselItems = select('#heroCarousel .carousel-item', true)

  heroCarouselItems.forEach((item, index) => {
    (index === 0) ?
    heroCarouselIndicators.innerHTML += "<li data-bs-target='#heroCarousel' data-bs-slide-to='" + index + "' class='active'></li>":
      heroCarouselIndicators.innerHTML += "<li data-bs-target='#heroCarousel' data-bs-slide-to='" + index + "'></li>"
  });

 
 

 

  /**
   * Initiate gallery lightbox 
   */
  const galleryLightbox = GLightbox({
    selector: '.gallery-lightbox'
  });
  

 

  /**
   * Animation on scroll
   */
  window.addEventListener('load', () => {
    AOS.init({
      duration: 1000,
      easing: 'ease-in-out',
      once: true,
      mirror: false
    })
  });

  /**
   * Initiate Pure Counter 
   */
  new PureCounter();

})()

// Existing JavaScript code in script.js

// Function to initialize your existing JavaScript code
function init() {
    // Add your existing JavaScript code here
    console.log('Initializing your existing JavaScript code...');
    // Example: Add event listeners, functions, etc.
}

// Function to handle image preview
function previewImage(event) {
    var reader = new FileReader(); // Create a FileReader object

    reader.onload = function() {
        var imgElement = document.createElement('img'); // Create img element
        imgElement.src = reader.result; // Set image source to the loaded file
        imgElement.style.maxWidth = '100%'; // Limit width to prevent overflow
        document.getElementById('imagePreview').innerHTML = ''; // Clear previous preview
        document.getElementById('imagePreview').appendChild(imgElement); // Append image to preview container
        document.querySelector('.image-section').style.display = 'block'; // Show image section
    };

    // Read the selected file as Data URL
    reader.readAsDataURL(event.target.files[0]);
}

// Function to handle image prediction (example)
function predictImage() {
    // This function can be used to handle image prediction logic (e.g., using AJAX to send image data to server)
    // Implement your prediction logic here
    alert('Implement your image prediction logic here!');
}

// Event listener to initialize the script after the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    init(); // Call your existing initialization function
    
    // Add event listeners for new functionalities (e.g., image preview)
    var uploadInput = document.getElementById('imageUpload');
    if (uploadInput) {
        uploadInput.addEventListener('change', previewImage);
    }

    var predictButton = document.getElementById('btn-predict');
    if (predictButton) {
        predictButton.addEventListener('click', predictImage);
    }
});

function predict() {
            var form_data = new FormData();
            var fileInput = document.getElementById('imageUpload');
            var file = fileInput.files[0];
            form_data.append('image', file);

            // Show loading animation
            document.getElementById('predictionResult').innerText = 'Loading...';

            // Make prediction by calling API /predict
            fetch('/predict', {
                method: 'POST',
                body: form_data,
            })
            .then(response => response.text())
            .then(data => {
                // Display the prediction result
                document.getElementById('predictionResult').innerText = 'Result: ' + data;
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('predictionResult').innerText = 'Error occurred.';
            });
        }