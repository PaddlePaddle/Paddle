let screenWidth = window.innerWidth;
let screenHeight = window.innerHeight;

let preloader = document.querySelector('#preloader');
let loader = document.querySelector('.loader');
preloader.style.width = screenWidth + "px";
preloader.style.height = screenHeight + "px";
loader.style.display = 'block';

window.onload = function() {
    preloader.style.display = "none";
};