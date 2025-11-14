import http from 'k6/http';
import { sleep, check } from 'k6';

export let options = {
    vus: 300000,
    duration: "1s",
  };
  
export default function () {
    let res = http.get("http://localhost:5000/compute");
    
    check(res, {
        "status is 200": (r) => r.status === 200,
        "response not empty": (r) => r.body.length > 0,
    });

    sleep(1);
}
