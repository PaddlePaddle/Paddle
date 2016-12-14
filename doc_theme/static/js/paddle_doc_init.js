$(document).ready(function(){
    $('.local-toc').on('click' ,'a.reference.internal', function (){
        $('.local-toc li.active').removeClass('active');
        $(this).parent('li').addClass('active');
    });

    if ($('.local-toc a:visible').length) {
        $('.local-toc > ul').addClass('nav nav-stacked');
        $('#doc-content').scrollspy({
            target: '.local-toc'
        });
		$('.local-toc').perfectScrollbar();
    } else {
		$('.doc-content-wrap').css('margin-left', '-=50px');
        $('.local-toc').remove();
    }

    if (!$('.doc-menu-vertical > ul > li.current > ul').length) {
        $('.doc-content-wrap').css('margin-left', '-=240px');
        $('.doc-menu-vertical').remove();
        $('.local-toc').css('left', '0');
    }

	$('.doc-menu-vertical .toctree-l2').each(function (i, e){
        $(e).toggleClass('has-child', !!$(e).find('ul').length);
    });

    $('.doc-menu-vertical').find('li.current').last().addClass('active');

    $('.doc-menu-vertical').perfectScrollbar();
});
