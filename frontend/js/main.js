/**
 * FaceAttend - Main JavaScript
 * Core functionality for Face Recognition Attendance System
 */

// ============================================================================
// NAVIGATION
// ============================================================================

document.addEventListener('DOMContentLoaded', function () {
    // Mobile menu toggle
    const mobileToggle = document.getElementById('mobileToggle');
    const navLinks = document.getElementById('navLinks');

    if (mobileToggle && navLinks) {
        mobileToggle.addEventListener('click', function () {
            navLinks.classList.toggle('active');
            mobileToggle.textContent = navLinks.classList.contains('active') ? '✕' : '☰';
        });
    }

    // Close mobile menu on link click
    const navLinkItems = document.querySelectorAll('.nav-link');
    navLinkItems.forEach(link => {
        link.addEventListener('click', () => {
            if (navLinks) navLinks.classList.remove('active');
            if (mobileToggle) mobileToggle.textContent = '☰';
        });
    });

    // Initialize live clock
    updateClock();
    setInterval(updateClock, 1000);

    // Initialize animations
    initAnimations();
});

// ============================================================================
// LIVE CLOCK
// ============================================================================

function updateClock() {
    const clockElement = document.getElementById('liveClock');
    if (clockElement) {
        const now = new Date();
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        clockElement.textContent = `${hours}:${minutes}:${seconds}`;
    }
}

// ============================================================================
// ANIMATIONS
// ============================================================================

function initAnimations() {
    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-fadeIn');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe cards and sections
    const animateElements = document.querySelectorAll('.card, .stat-card, .section-title');
    animateElements.forEach(el => observer.observe(el));
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Format a date string for display
 * @param {string|Date} date - Date to format
 * @returns {string} Formatted date string
 */
function formatDate(date) {
    const d = new Date(date);
    return d.toLocaleDateString('en-IN', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

/**
 * Format time for display
 * @param {string|Date} date - Date to format
 * @returns {string} Formatted time string
 */
function formatTime(date) {
    const d = new Date(date);
    return d.toLocaleTimeString('en-IN', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
    });
}

/**
 * Get initials from a name
 * @param {string} name - Full name
 * @returns {string} Initials (max 2 characters)
 */
function getInitials(name) {
    const names = name.trim().split(' ');
    if (names.length >= 2) {
        return names[0][0] + names[names.length - 1][0];
    }
    return names[0].substring(0, 2);
}

/**
 * Show a toast notification
 * @param {string} message - Message to display
 * @param {string} type - Type of notification (success, error, warning, info)
 */
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    padding: 1rem 1.5rem;
    background: var(--color-bg-card);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    color: var(--color-text-primary);
    box-shadow: var(--shadow-lg);
    z-index: 1000;
    animation: slideIn 0.3s ease;
  `;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'fadeOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

/**
 * Debounce function for performance optimization
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ============================================================================
// STATISTICS UPDATE (Mock Data)
// ============================================================================

function updateDashboardStats() {
    // This would normally fetch from an API
    const stats = {
        totalEnrolled: Math.floor(Math.random() * 10) + 45,
        presentToday: Math.floor(Math.random() * 10) + 28,
        lateArrivals: Math.floor(Math.random() * 8),
        avgAccuracy: Math.floor(Math.random() * 5) + 95
    };

    const totalEnrolled = document.getElementById('totalEnrolled');
    const presentToday = document.getElementById('presentToday');
    const lateArrivals = document.getElementById('lateArrivals');
    const avgAccuracy = document.getElementById('avgAccuracy');

    if (totalEnrolled) totalEnrolled.textContent = stats.totalEnrolled;
    if (presentToday) presentToday.textContent = stats.presentToday;
    if (lateArrivals) lateArrivals.textContent = stats.lateArrivals;
    if (avgAccuracy) avgAccuracy.textContent = stats.avgAccuracy + '%';
}

// Update stats every 30 seconds (for demo)
setInterval(updateDashboardStats, 30000);

// ============================================================================
// KEYBOARD SHORTCUTS
// ============================================================================

document.addEventListener('keydown', function (e) {
    // Alt + 1-4 for navigation
    if (e.altKey) {
        switch (e.key) {
            case '1':
                window.location.href = 'index.html';
                break;
            case '2':
                window.location.href = 'camera.html';
                break;
            case '3':
                window.location.href = 'attendance.html';
                break;
            case '4':
                window.location.href = 'enroll.html';
                break;
        }
    }
});

// ============================================================================
// SERVICE WORKER REGISTRATION (Optional - for PWA)
// ============================================================================

// if ('serviceWorker' in navigator) {
//   window.addEventListener('load', () => {
//     navigator.serviceWorker.register('/sw.js')
//       .then(registration => console.log('SW registered'))
//       .catch(error => console.log('SW registration failed:', error));
//   });
// }

console.log('🎉 FaceAttend loaded successfully!');
