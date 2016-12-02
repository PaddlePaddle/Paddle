$(document).ready(function(){
    $('.local-toc').on('click' ,'a.reference.internal', function (){
        $('.local-toc li.active').removeClass('active');
        $(this).parent('li').addClass('active');
    });
    
    
    if (!$('.doc-menu-vertical > ul > li.current > ul').length) {
        $('.doc-content-wrap').css('margin-left', '-=240px');
        $('.doc-menu-vertical').remove();
        $('.local-toc').css('left', '0');
    }
    $('.doc-menu-vertical .toctree-l2').each(function (i, e){
        $(e).toggleClass('has-child', !!$(e).find('ul').length);
    });
    if ($('.local-toc a:visible').length) {
        $('.doc-content-wrap').css('margin-left', '+=50px');
        $('.local-toc > ul').addClass('nav nav-stacked');
        $('#doc-content').scrollspy({
            target: '.local-toc'
        });
    } else {
        $('.local-toc').remove();
    }

    $('.doc-menu-vertical').find('li.current').last().addClass('active');

    $('.doc-menu-vertical').perfectScrollbar();
    $('.local-toc').perfectScrollbar();
});
