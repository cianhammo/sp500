// shared-header.js
document.addEventListener('DOMContentLoaded', function() {
    // Create the header element
    const header = document.createElement('header');
    header.className = 'fixed top-0 left-0 right-0 bg-black/80 backdrop-blur-sm border-b border-white/10 z-50';
    
    // Get current page from pathname
    const currentPath = window.location.pathname;
    const currentPage = currentPath.split('/').pop().split('.')[0].toLowerCase();
    console.log('Current path:', currentPath);
    console.log('Current page:', currentPage);
    
    // Generate the header HTML
    header.innerHTML = `
        <nav class="max-w-4xl mx-auto px-4">
            <div class="flex items-center justify-center h-16">
                <div class="flex items-center space-x-8">
                    <a 
                        href="/SEC.html"
                        style="background-color: ${currentPage === 'sec' ? 'rgba(255, 255, 255, 0.1)' : 'transparent'}"
                        class="flex items-center space-x-8 px-3 py-2 text-lg font-medium transition-colors no-underline hover:underline ${
                            currentPage === 'sec' ? 'text-white' : 'text-gray-300 hover:text-white'
                        }"
                    >
                        <span>SEC Filings</span>
                    </a>

                    <a 
                        href="/index.html"
                        style="background-color: ${currentPage === 'index' ? 'rgba(255, 255, 255, 0.1)' : 'transparent'}"
                        class="flex items-center space-x-2 px-3 py-2 text-lg font-medium transition-colors no-underline hover:underline ${
                            currentPage === 'index' ? 'text-white' : 'text-gray-300 hover:text-white'
                        }"
                    >
                        <span>S&P 500</span>
                    </a>

                    <a 
                        href="/about.html"
                        style="background-color: ${currentPage === 'about' ? 'rgba(255, 255, 255, 0.1)' : 'transparent'}"
                        class="flex items-center space-x-2 px-3 py-2 text-lg font-medium transition-colors no-underline hover:underline ${
                            currentPage === 'about' ? 'text-white' : 'text-gray-300 hover:text-white'
                        }"
                    >
                        <span>About</span>
                    </a>
                </div>
            </div>
        </nav>
    `;

    // Add some basic styles for the navigation icons
    const style = document.createElement('style');
    style.textContent = `
        /* Add margin to the main content to prevent it from being hidden under the header */
        body {
            margin-top: 64px;
        }

        /* Set header background opacity while keeping text opaque */
        header {
            background-color: rgba(0, 0, 0, 0.5) !important;
            border: none !important;
        }
        
        /* Make header text fully opaque and consistent size */
        header a {
            font-size: 1.25rem !important;
            text-decoration: none !important;
            border-radius: 4px;
            transition: all 0.2s ease;
        }
        
        header a:hover {
            text-decoration: underline !important;
        }
        
        /* Responsive adjustments */
        @media (max-width: 480px) {
            .space-x-8 > * + * {
                margin-left: 1rem;
            }
        }
    `;
    document.head.appendChild(style);

    // Insert the header at the start of the body
    document.body.insertBefore(header, document.body.firstChild);
});