/*****************************************************************************
 *
 * ALPS MPS DMRG Project
 *
 * Copyright (C) 2014 Institute for Theoretical Physics, ETH Zurich
 *               2011-2011 by Bela Bauer <bauerb@phys.ethz.ch>
 *
 * This software is part of the ALPS Applications, published under the ALPS
 * Application License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 *
 * You should have received a copy of the ALPS Application License along with
 * the ALPS Applications; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#ifndef UTILS_RESULTS_COLLECTOR_H
#define UTILS_RESULTS_COLLECTOR_H

#include <boost/any.hpp>
#include <vector>
#include <memory>

class results_collector
{
private:

    class collector_impl_base;
    template <class T> class collector_impl;

public:

    class collector_proxy {
    typedef std::shared_ptr<results_collector::collector_impl_base> coll_type;
    public:
        collector_proxy(coll_type & collector_)
        : collector(collector_)
        { }

        template<class T>
        void operator<<(T const& val);

        // Needed for a dirty hack for loading iteration_results
        template<class T>
        void new_collector();

        template<class T>
        void operator>>(T const& val);

        const std::vector<boost::any>& get() const;

    private:
        coll_type & collector;
    };

    collector_proxy operator[] (std::string name);

    void clear();

    template <class Archive>
    void save(Archive & ar) const;

    template <class Archive>
    void load(Archive & ar);

    bool empty() const ;
    bool has(const std::string key) const {return collection.find(key) != collection.end(); }
private:
    std::map<std::string, std::shared_ptr<collector_impl_base> > collection;
};

#endif
