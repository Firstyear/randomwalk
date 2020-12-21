use mathru::statistics::distrib::{Normal, Distribution};
use rayon::prelude::*;

// Rather than using an array for each parameter/result IE
//
// latencies: Vec<usize> 
// response: Vec<bool>,
// evidence: Vec<Vec<f64>,
//
// We create a result type that keeps the related values/information closer
// in memory. This reduces memory fragmentation (since we are now in a single
// array, rather than multiple), and generally just makes your cpu caches
// happier.
#[derive(Debug)]
struct ModelResult {
    latency: usize,
    response: bool,
    evidence: Vec<f64>,
}

fn execute_model(
    // How many iterations to perform
    num_reps: usize,
    // Upper bound on how many samples before we declare this attempt a failure, and skip.
    max_samples: usize,
    // % amount of evidence that is avaliable during sampling (the higher the drift rate the larger the "steps"). 
    drift: f64,
    // %sdrw is the amount of noise that exists, using a standard deviation distribution from which we will sample the evidence. 
    sdrw: f64,
    // %the decision threshhold is reached once the drift rate reaches 3. aka The distance between the two boundaries 
    criterion: f64,
) -> Result<Vec<ModelResult>, ()> {

    let distrib: Normal<f64> = Normal::new(drift, sdrw);

    // This creates a thread pool, and runs each "attempt" on a different CPU core. Because there
    // is no cross-talk/relationship between samples, these are all calculated indepedently, this
    // gives us great ability to parallelise this problem in this way. After we have done all the
    // work in parallel we collect all the results into a single array for later processing/analysis.
    //
    // Due to how rust works/optimises, the results array is correctly pre-allocated to the size
    // of reps here, which means that we avoid costly reallocs.
    let results: Vec<ModelResult> = (0..num_reps)
        .into_par_iter()
        .filter_map(|_i| {

        let mut acc: f64 = 0.0;
        let mut evidence: Vec<f64> = Vec::with_capacity(max_samples);

        // Rather than generate all the samples, then walk through them to find the point at which
        // we have reached the decision threshold, we generate each sample one at a time and
        // continue to process and accumlate that, shortcutting (early-return) when we have
        // passed the criterion (decision point).
        //
        for latency in 0..max_samples {
            // %generated a distribution of randomly sampled evidence with a mean of drift and standard deviation of sdrw
            let v = distrib.random();
            // %the accumulation of that evidence is calculated
            // evidence(i,:) = cumsum([0 genSample]);
            acc = acc + v;
            evidence.push(acc);

            // %calculate p, the first value to reach the decision threshold. 
            //     p = find((abs(evidence(i,:)) > criterion),1); 
            if acc.abs() > criterion {
                let response = acc.is_sign_positive();
                // Complete, build the result. Wrapping in the "Some" variant for Option
                // indicates to the iterator that we succedded and that we should keep this
                // valid result.
                return Some(ModelResult {
                    latency,
                    response,
                    evidence,
                });
            }
        }

        // If we were unable to get enough samples, log an error message to the display,
        // and return "None" - this is filtered out in the parallel iterator, which allows
        // us to reject invalid attempts (at the cost that we will always proceduce reps or
        // fewer results).
        //
        // However, due to the shortcut return process, we can raise samples to a very large
        // amount, making this highly improbable to reach :)
        eprintln!("Threshold not met -> pls give more samps");
        None
    })
    .collect();

    // 🎉🎉🎉
    Ok(results)
}


fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn do_the_thang() {
        // let reps = 10;
        // let bound = 1000;
        let reps = 1_000_000;
        let bound = 10000;

        let res = execute_model(
            reps,
            bound,
            0.1,
            0.3,
            3.0,
        )
        .expect("Failed to run model");
        eprintln!("successful samples -> {:?}", res.len());
    }
}
