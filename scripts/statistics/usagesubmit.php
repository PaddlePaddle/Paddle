<?php

    include ("config.php");

    function exec_sql($sql)
    {
     
        global $host;
        global $port;
        global $dbname;
        global $credentials;

        //echo "$host $port $dbname $credentials";
        $db = pg_connect( "$host $port $dbname $credentials"  );
        if(!$db){
           echo "Error : Unable to open database\n";
        }
        
        //$sql = "select * from download_info;";
    
        $ret = pg_query($db, $sql);
        if(!$ret){
           echo pg_last_error($db);
           return False;
        } 
    
    
        $arr = array();
        if (stripos(trim($sql), 'select') === 0)
        {
            while($row = pg_fetch_row($ret)){
                $arr[] = $row;
            }
    
        }
        else
        {
            $arr = True;
        }
        pg_close($db);
    
        return $arr;
    }
    
    function gen_return_data($status, $msg = '', $data = '')
    {
        $return_msg=array();
        $return_msg['status'] = max(0,intval($status));
        $return_msg['msg']=$msg;
        if ($data != '')
        {
            $return_msg['data']=$data;
        }
        return json_encode($return_msg);
    }
    
    function request_check($ip)
    {
    
        //1s内最大访问次数为5次
    
        $count_flag = $ip . '_count';
        $time_flag = $ip . '_time';
    
        //session_start();
        $sec = date('s');
        if (! isset($_SESSION[$count_flag])) $_SESSION[$count_flag] = 0;
        if (! isset($_SESSION[$time_flag])) $_SESSION[$time_flag] =  $sec;
        if ($_SESSION[$time_flag] !== $sec){//每秒钟重置一次
            $_SESSION[$count_flag] = 0;
            $_SESSION[$time_flag] =  $sec;
        }
    
        //小于5次就执行
        if ($_SESSION[$count_flag] < 5){
            $_SESSION[$count_flag] = $_SESSION[$count_flag] + 1;
            return True;
        }
        else
        {
            return False;
        }
    }
    
    function get_ip()
    {
        $onlineip = '';
        //获得客户端ip
        $cip = getenv('HTTP_CLIENT_IP');
        $xip = getenv('HTTP_X_FORWARDED_FOR');
        $rip = getenv('REMOTE_ADDR');
        $srip = $_SERVER['REMOTE_ADDR'];
        if($cip && strcasecmp($cip, 'unknown')) {
            $onlineip = $cip;
        } elseif($xip && strcasecmp($xip, 'unknown')) {
            $onlineip = $xip;
        } elseif($rip && strcasecmp($rip, 'unknown')) {
            $onlineip = $rip;
        } elseif($srip && strcasecmp($srip, 'unknown')) {
            $onlineip = $srip;
        }
        preg_match("/[\d\.]{7,15}/", $onlineip, $match);
        $onlineip = $match[0] ? $match[0] : 'unknown';
    
        return $onlineip;
    }
    
    $ip = get_ip();
    $check_res = request_check($ip);
    if(! $check_res)
    {
        echo gen_return_data(1, 'fail: query exceed!');
        exit(0);
    }
    
    $type = max(0, intval($_POST['type']));
    $time = strftime('%Y-%m-%d %H:%M:%S', strtotime($_POST['time']));
    
    $payload = json_encode(json_decode($_POST['payload']));
    if (!$payload)
    {
        echo gen_return_data(1, 'fail: invalid payload');
        exit(0);
    }
    
    //echo $ip, $type, $time, $payload;
    
    $sql = "insert into usage_info(ip, type, time, payload, create_time) 
            values('$ip', $type, '$time', '$payload', now());";
    if(!exec_sql($sql))
    {
        echo gen_return_data(1, 'update record failed!');
        exit(0);
    }
    else
    {
        echo gen_return_data(0, 'success');
        exit(0);
    }


?>
